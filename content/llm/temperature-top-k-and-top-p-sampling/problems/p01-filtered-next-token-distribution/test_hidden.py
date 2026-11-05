# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from submission import filtered_distribution as _learner


def _call(*args, **kwargs):
    """Call the learner; an exception on a valid input is reported as a named contract failure."""
    try:
        return _learner(*args, **kwargs)
    except ValueError:
        pytest.fail("valid arguments were rejected with ValueError: check the argument rules")
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _expect_value_error(*args, **kwargs):
    try:
        _learner(*args, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid argument was accepted instead of raising ValueError")


def _reference(logits, temperature, top_k=None, top_p=None):
    """Pure-Python reference: temperature, softmax, top-k then top-p, with explicit tie-breaks."""
    z = [x / temperature for x in logits]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    prob = [v / total for v in e]
    V = len(prob)
    by_rank = sorted(range(V), key=lambda i: (-prob[i], i))  # ties: lower index first
    keep = set(range(V))
    if top_k is not None and top_k < V:
        keep = set(by_rank[:top_k])
    if top_p is not None and top_p < 1.0:
        ranked = [i for i in by_rank if i in keep]
        mass = sum(prob[i] for i in keep)
        chosen: set[int] = set()
        running = 0.0
        for i in ranked:
            chosen.add(i)
            running += prob[i] / mass
            if running >= top_p:
                break
        keep = chosen
    kept_mass = sum(prob[i] for i in keep)
    return [prob[i] / kept_mass if i in keep else 0.0 for i in range(V)]


def test_random_logits_match_reference_across_filter_combinations():
    gen = torch.Generator().manual_seed(2024)
    combos = [
        (1.0, None, None),
        (0.7, 3, None),
        (1.3, None, 0.8),
        (0.5, 2, 0.9),
        (1.0, 5, 0.6),
        (2.0, 1, None),
        (1.0, 100, 0.95),
    ]
    for trial in range(10):
        logits = torch.randn(8, generator=gen, dtype=torch.float64) * 2.0
        for temperature, top_k, top_p in combos:
            got = _call(logits, temperature, top_k=top_k, top_p=top_p)
            want = torch.tensor(
                _reference(logits.tolist(), temperature, top_k, top_p), dtype=torch.float64
            )
            torch.testing.assert_close(
                got,
                want,
                atol=1e-9,
                rtol=1e-9,
                msg=(
                    f"filtered distribution differs from the reference (trial {trial}): "
                    "check the order: temperature, softmax, top-k with renormalisation, then top-p "
                    "over the sorted survivors (cumulative sum of renormalised probabilities), "
                    "then a final renormalisation"
                ),
            )


def test_top_p_exact_boundary_keeps_the_token_that_reaches_p():
    # Uniform logits give probabilities of exactly 0.25, so the cumulative sums are exact.
    logits = torch.zeros(4, dtype=torch.float64)
    out = _call(logits, 1.0, top_p=0.5)
    assert out.tolist() == [0.5, 0.5, 0.0, 0.0], (
        "top-p boundary off by one: the token that makes the cumulative sum reach p must be kept"
    )


def test_top_p_exact_boundary_does_not_keep_an_extra_token():
    logits = torch.zeros(8, dtype=torch.float64)  # each probability is exactly 0.125
    out = _call(logits, 1.0, top_p=0.375)
    kept = [i for i, v in enumerate(out.tolist()) if v > 0]
    assert kept == [0, 1, 2], "top-p kept an extra token past the boundary: check >= versus >"


def test_top_p_acts_on_the_top_k_survivors_after_renormalising():
    # Probabilities 0.4, 0.3, 0.2, 0.1. Top-k=2 keeps 0.4 and 0.3, renormalised to 4/7 = 0.571.
    # That first survivor alone reaches top_p=0.5, so only one token remains. Using the
    # unrenormalised 0.4 and 0.7 would keep two tokens instead.
    logits = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64))
    got = _call(logits, 1.0, top_k=2, top_p=0.5)
    want = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-9, rtol=0), (
        "top-p must act on the renormalised top-k survivors, not the unrenormalised ones"
    )


def test_top_p_ties_keep_the_lowest_indices_first():
    out = _call(torch.zeros(4, dtype=torch.float64), 1.0, top_p=0.25)
    assert out.tolist() == [1.0, 0.0, 0.0, 0.0], "tied probabilities must be ordered by lower index"


def test_wide_tie_block_keeps_the_lowest_indices_for_top_k():
    # 300 equal logits: every probability is identical, so only the tie-break decides which five
    # survive. The contract says the five lowest indices. (Unstable sorts pass on short inputs.)
    out = _call(torch.zeros(300, dtype=torch.float64), 1.0, top_k=5)
    kept = [i for i, v in enumerate(out.tolist()) if v > 0]
    assert kept == [0, 1, 2, 3, 4], (
        "top-k tie-break must keep the lowest indices in a wide tie block"
    )


def test_wide_tie_block_keeps_the_lowest_indices_for_top_p():
    # 16 equal probabilities of exactly 1/16: top_p = 0.5 is exactly the first eight tokens.
    out = _call(torch.zeros(16, dtype=torch.float64), 1.0, top_p=0.5)
    kept = [i for i, v in enumerate(out.tolist()) if v > 0]
    assert kept == list(range(8)), "top-p tie-break must order equal probabilities by lower index"


def test_tiny_top_p_still_keeps_the_best_token():
    gen = torch.Generator().manual_seed(9)
    logits = torch.randn(6, generator=gen, dtype=torch.float64)
    out = _call(logits, 1.0, top_p=1e-12)
    best = int(torch.argmax(logits).item())
    assert out.argmax().item() == best and out[best].item() == pytest.approx(1.0), (
        "top-p must keep at least one token: the best one, even when p is tiny"
    )


def test_top_p_one_leaves_every_token_in_play():
    logits = torch.tensor([0.0, -40.0, 0.0], dtype=torch.float64)
    out = _call(logits, 1.0, top_p=1.0)
    assert (out > 0).all().item(), (
        "top_p = 1 must be a no-op, but a tiny-probability token was removed"
    )


def test_top_k_at_least_vocabulary_size_is_a_no_op():
    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(5, generator=gen, dtype=torch.float64)
    plain = _call(logits, 1.0)
    for k in (5, 6, 1000):
        torch.testing.assert_close(
            _call(logits, 1.0, top_k=k),
            plain,
            atol=1e-12,
            rtol=1e-12,
            msg="top_k >= V must remove nothing: check the top-k branch bound",
        )


def test_top_k_one_is_a_one_hot_at_the_argmax():
    gen = torch.Generator().manual_seed(6)
    logits = torch.randn(7, generator=gen, dtype=torch.float64)
    out = _call(logits, 0.9, top_k=1)
    expected = torch.zeros(7, dtype=torch.float64)
    expected[int(torch.argmax(logits).item())] = 1.0
    torch.testing.assert_close(
        out, expected, atol=1e-12, rtol=0, msg="top_k=1 must be one-hot at the argmax"
    )


def test_temperature_one_is_the_plain_softmax():
    gen = torch.Generator().manual_seed(8)
    logits = torch.randn(6, generator=gen, dtype=torch.float64)
    want = torch.tensor(_reference(logits.tolist(), 1.0), dtype=torch.float64)
    torch.testing.assert_close(
        _call(logits, 1.0),
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="temperature 1 must give the plain softmax",
    )


def test_small_temperature_sharpens_without_overflow():
    logits = torch.tensor([10.0, 9.5, -300.0], dtype=torch.float64)
    out = _call(logits, 1e-3)
    assert torch.isfinite(out).all().item(), (
        "small temperature overflowed: subtract the max before exp"
    )
    assert out.tolist()[0] == pytest.approx(1.0, abs=1e-9), (
        "small temperature must concentrate on the best token"
    )


def test_large_logit_magnitudes_stay_finite_and_normalised():
    gen = torch.Generator().manual_seed(12)
    logits = torch.randn(9, generator=gen, dtype=torch.float64) * 1000.0
    out = _call(logits, 1.0, top_p=0.9)
    assert torch.isfinite(out).all().item(), (
        "large logits produced non-finite probabilities: stabilise the softmax"
    )
    assert out.sum().item() == pytest.approx(1.0, abs=1e-12), "probabilities must sum to 1"


def test_single_entry_vocabulary_returns_probability_one():
    out = _call(torch.tensor([3.5], dtype=torch.float64), 0.7, top_k=1, top_p=0.2)
    assert out.tolist() == [1.0], "a one-token vocabulary must return probability 1"


def test_output_is_float64_non_negative_and_sums_to_one():
    gen = torch.Generator().manual_seed(21)
    for _ in range(5):
        logits = torch.randn(10, generator=gen, dtype=torch.float64)
        out = _call(logits, 1.1, top_k=4, top_p=0.85)
        assert out.dtype == torch.float64, "output must be float64"
        assert (out >= 0).all().item(), "probabilities must be non-negative"
        assert out.sum().item() == pytest.approx(1.0, abs=1e-12), "probabilities must sum to 1"


def test_logits_are_not_mutated():
    logits = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    before = logits.clone()
    _call(logits, 0.5, top_k=2, top_p=0.9)
    assert torch.equal(logits, before), "logits were modified in place: divide into a new tensor"


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, float("nan"), float("inf"), True],
    ids=["zero", "negative", "nan", "infinite", "bool"],
)
def test_bad_temperature_raises_value_error(temperature):
    _expect_value_error(torch.zeros(3, dtype=torch.float64), temperature)


@pytest.mark.parametrize("top_k", [0, -2, True, 1.0], ids=["zero", "negative", "bool", "float"])
def test_bad_top_k_raises_value_error(top_k):
    _expect_value_error(torch.zeros(3, dtype=torch.float64), 1.0, top_k=top_k)


@pytest.mark.parametrize(
    "top_p",
    [0.0, -0.1, 1.5, float("nan"), True],
    ids=["zero", "negative", "above_one", "nan", "bool"],
)
def test_bad_top_p_raises_value_error(top_p):
    _expect_value_error(torch.zeros(3, dtype=torch.float64), 1.0, top_p=top_p)


def test_two_dimensional_logits_raise_value_error():
    _expect_value_error(torch.zeros(2, 3, dtype=torch.float64), 1.0)


def test_float32_logits_raise_value_error():
    _expect_value_error(torch.zeros(3, dtype=torch.float32), 1.0)


def test_non_finite_logits_raise_value_error():
    _expect_value_error(torch.tensor([0.0, float("inf")], dtype=torch.float64), 1.0)


def test_top_p_just_below_one_still_returns_a_distribution():
    # Float64 cumulative sums can finish a hair below top_p. A valid call must still return a
    # distribution that sums to one, never raise.
    gen = torch.Generator().manual_seed(31)
    for _ in range(40):
        logits = torch.randn(12, generator=gen, dtype=torch.float64) * 3.0
        for top_p in (1 - 2.0**-53, 1 - 2.0**-52, 1 - 2.0**-51, 1 - 2.0**-50):
            out = _call(logits, 1.0, top_p=top_p)
            if abs(out.sum().item() - 1.0) > 1e-12 or (out < 0).any().item():
                pytest.fail("a top_p just below one returned something that is not a distribution")
