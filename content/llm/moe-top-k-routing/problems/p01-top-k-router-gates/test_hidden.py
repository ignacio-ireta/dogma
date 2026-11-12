# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of
# mistake and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from submission import top_k_route as _learner

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def _real_values(gen, shape, dtype=torch.float64):
    """Real numbers in the shape asked for: standardised breast_cancer measurements.

    The tensors of this problem are weights, gradients, scores or hidden states. No dataset
    holds those, so real measurements stand in for them: skewed, correlated along a row, with
    a few entries far from the rest. The torch generator picks the draw.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = shape[0]
    shape = tuple(int(side) for side in shape)
    total = math.prod(shape)
    if total == 0:
        return torch.zeros(shape, dtype=dtype)
    draw = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    key = f"p01-top-k-router-gates:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def _call(logits, k):
    """Call the learner; an exception on a valid input is reported as a contract failure."""
    try:
        return _learner(logits, k)
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _invalid(exc_type, logits, k):
    try:
        _learner(logits, k)
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument raised {type(exc).__name__}, expected {exc_type.__name__}")
    pytest.fail(f"invalid argument was accepted instead of raising {exc_type.__name__}")


def _reference(rows, num_experts, k):
    """Pure-Python route. The gates come from the full-row probabilities, renormalised over the
    chosen experts. That is a different formula from the logit-only softmax the solution uses."""
    idx_rows, gate_rows = [], []
    counts = [0] * num_experts
    for row in rows:
        top = max(row)
        exps = [math.exp(x - top) for x in row]
        probs = [e / sum(exps) for e in exps]
        chosen = sorted(range(num_experts), key=lambda e: (-row[e], e))[:k]
        mass = sum(probs[e] for e in chosen)
        idx_rows.append(chosen)
        gate_rows.append([probs[e] / mass for e in chosen])
        for e in chosen:
            counts[e] += 1
    return idx_rows, gate_rows, counts


def _check_against_reference(logits, k):
    rows = logits.tolist()
    num_experts = logits.shape[1]
    want_idx, want_gates, want_counts = _reference(rows, num_experts, k)
    idx, gates, counts = _call(logits, k)
    assert idx.tolist() == want_idx, (
        "selected experts differ from the k largest logits: check the ranking and the tie rule"
    )
    torch.testing.assert_close(
        gates,
        torch.tensor(want_gates, dtype=torch.float64).reshape(logits.shape[0], k),
        atol=1e-9,
        rtol=1e-9,
        msg="gates differ from the renormalised chosen probabilities: check the normalisation",
    )
    assert counts.tolist() == want_counts, (
        "expert_counts disagree with the routing: count every (token, slot) pair"
    )


def test_seeded_random_routes_match_probability_reference():
    gen = torch.Generator().manual_seed(101)
    for tokens, experts, k in [(5, 4, 2), (7, 6, 3), (3, 8, 1)]:
        logits = _real_values(gen, [tokens, experts], torch.float64) * 2.0
        _check_against_reference(logits, k)


def test_tie_at_the_k_boundary_keeps_lowest_indices():
    logits = torch.tensor([[1.0, 3.0, 3.0, 3.0, 0.0]], dtype=torch.float64)
    idx, gates, _ = _call(logits, 2)
    assert idx.tolist() == [[1, 2]], (
        "a tie at the k-th place must keep the lowest expert indices: check the tie order"
    )
    torch.testing.assert_close(
        gates,
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="tied chosen experts must get equal gates",
    )


def test_gates_renormalise_over_chosen_experts_only():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float64)
    idx, gates, _ = _call(logits, 2)
    assert idx.tolist() == [[0, 1]], "wrong experts chosen"
    assert abs(float(gates.sum()) - 1.0) < 1e-12, (
        "gates must be renormalised over the k chosen experts: each row sums to 1"
    )
    torch.testing.assert_close(
        gates,
        torch.softmax(torch.tensor([[2.0, 1.0]], dtype=torch.float64), dim=-1),
        atol=1e-9,
        rtol=1e-9,
        msg="gate values must equal the softmax over the chosen logits",
    )


def test_counts_include_every_slot_not_only_the_first():
    logits = torch.tensor(
        [[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0], [3.0, 0.0, 2.0, 1.0]], dtype=torch.float64
    )
    _, _, counts = _call(logits, 2)
    assert counts.tolist() == [2, 1, 2, 1], (
        "expert_counts must count the second slot too: the total must be T * k"
    )
    assert int(counts.sum()) == 6, "expert_counts must sum to T * k"


def test_large_logit_magnitudes_do_not_overflow():
    gen = torch.Generator().manual_seed(102)
    logits = _real_values(gen, [4, 5], torch.float64) * 400.0
    _, gates, _ = _call(logits, 2)
    assert torch.isfinite(gates).all(), (
        "large logits produced non-finite gates: subtract the max before exp"
    )
    _check_against_reference(logits, 2)


def test_selection_uses_logits_when_probabilities_underflow():
    # exp(-800) and exp(-801) both underflow to zero in float64, so probabilities cannot order
    # these two experts. The logits can: -800 beats -801.
    logits = torch.tensor([[0.0, -801.0, -800.0]], dtype=torch.float64)
    idx, gates, _ = _call(logits, 2)
    assert idx.tolist() == [[0, 2]], (
        "selection must rank the logits: ranking by underflowed probabilities breaks the order"
    )
    torch.testing.assert_close(
        gates,
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
        msg="gates must be the softmax of the chosen logits",
    )


def test_k_equal_to_num_experts_selects_every_expert():
    gen = torch.Generator().manual_seed(103)
    logits = _real_values(gen, [4, 3], torch.float64)
    idx, _gates, counts = _call(logits, 3)
    assert sorted(idx[0].tolist()) == [0, 1, 2], "with k = E every expert must be chosen once"
    _check_against_reference(logits, 3)
    assert counts.tolist() == [4, 4, 4], "with k = E every expert gets every token"


@pytest.mark.parametrize("k", [0, 5], ids=["zero", "above-num-experts"])
def test_k_outside_one_to_num_experts_raises_value_error(k):
    logits = torch.zeros(2, 4, dtype=torch.float64)
    _invalid(ValueError, logits, k)


def test_bool_k_raises_type_error():
    logits = torch.zeros(2, 4, dtype=torch.float64)
    _invalid(TypeError, logits, True)


def test_one_dimensional_logits_raise_value_error():
    _invalid(ValueError, torch.zeros(4, dtype=torch.float64), 2)


def test_float32_logits_raise_value_error():
    _invalid(ValueError, torch.zeros(2, 4, dtype=torch.float32), 2)


@pytest.mark.parametrize("bad", [math.nan, math.inf], ids=["nan", "inf"])
def test_non_finite_logit_raises_value_error(bad):
    logits = torch.zeros(2, 4, dtype=torch.float64)
    logits[1, 2] = bad
    _invalid(ValueError, logits, 2)


def test_output_dtypes_and_shapes_follow_the_contract():
    gen = torch.Generator().manual_seed(104)
    logits = _real_values(gen, [6, 5], torch.float64)
    idx, gates, counts = _call(logits, 2)
    assert idx.dtype == torch.int64, "expert_indices must be int64"
    assert gates.dtype == torch.float64, "gate_weights must be float64"
    assert counts.dtype == torch.int64, "expert_counts must be int64"
    assert tuple(idx.shape) == (6, 2), "expert_indices must have shape (T, k)"
    assert tuple(gates.shape) == (6, 2), "gate_weights must have shape (T, k)"
    assert tuple(counts.shape) == (5,), "expert_counts must have shape (E,)"


def test_inputs_are_not_mutated():
    gen = torch.Generator().manual_seed(105)
    logits = _real_values(gen, [5, 4], torch.float64)
    before = logits.clone()
    _call(logits, 2)
    assert torch.equal(logits, before), "the logits tensor must not be modified in place"


def test_tie_for_first_place_in_a_wide_router_keeps_lowest_index():
    # Every one of the 200 experts has the same logit, so all of them tie for first place.
    # The tie rule (ties go to the lowest index) must still hold when there are many experts.
    logits = torch.ones(1, 200, dtype=torch.float64)
    idx, _, _ = _call(logits, 1)
    assert idx.tolist() == [[0]], (
        "a tie for first place in a wide router must go to the lowest expert index: "
        "keep equal logits in their original order"
    )
