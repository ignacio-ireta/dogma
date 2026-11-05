# Hidden tests. The oracle is torch.nn.functional.cross_entropy with ignore_index (tests may use
# torch.nn; the learner may not). Messages name the kind of mistake, never hidden inputs or values.
from __future__ import annotations

import math

import torch
import torch.nn.functional as ref_f  # oracle only
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import masked_nll_and_perplexity

ATOL = 1e-9
RTOL = 1e-9
# float32 oracle comparison: float32 has ~1.2e-7 relative precision and a log-sum-exp over a few
# dozen classes accumulates a few ulps. 1e-5 does not fail correct float32 code, and it is still
# far below any real mistake (a wrong denominator, a naive exp).
ATOL_F32 = 1e-5
RTOL_F32 = 1e-5


def _call(logits, targets, ignore_index=-100):
    """Run the learner on a valid call. Failures are AssertionErrors that name the mistake."""
    snap_l, snap_t = logits.clone(), targets.clone()
    try:
        out = masked_nll_and_perplexity(logits, targets, ignore_index=ignore_index)
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: check the shapes, the target dtype and the "
            "ignore_index keyword"
        ) from None
    if not torch.equal(logits, snap_l) or not torch.equal(targets, snap_t):
        raise AssertionError("an input tensor was modified; compute from copies, never in place")
    if not isinstance(out, tuple) or len(out) != 2 or not all(torch.is_tensor(t) for t in out):
        raise AssertionError("return a tuple of two tensors: mean_nll and perplexity")
    mean_nll, ppl = out
    if mean_nll.ndim != 0 or ppl.ndim != 0:
        raise AssertionError("mean_nll and perplexity must be 0-dimensional tensors")
    if mean_nll.dtype != logits.dtype or ppl.dtype != logits.dtype:
        raise AssertionError("results must keep the dtype of the logits; do not downcast")
    return mean_nll, ppl


def _oracle(logits, targets, ignore_index=-100):
    """PyTorch cross-entropy (mean over non-ignored targets) and its perplexity."""
    vocab = logits.shape[-1]
    ce = ref_f.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=ignore_index, reduction="mean"
    )
    return ce, torch.exp(ce)


def _expect_value_error(thunk, message):
    raised = None
    try:
        thunk()
    except Exception as exc:
        raised = exc
    if raised is None:
        raise AssertionError(message)
    if not isinstance(raised, ValueError):
        raise AssertionError(f"invalid argument must raise ValueError, not {type(raised).__name__}")


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _random_batch(seed: int, b: int, t: int, v: int, *, dtype=torch.float64, ignore_rate=0.3):
    g = _gen(seed)
    logits = torch.randn((b, t, v), generator=g, dtype=dtype) * 2.0
    targets = torch.randint(0, v, (b, t), generator=g)
    drop = torch.rand((b, t), generator=g) < ignore_rate
    targets = targets.masked_fill(drop, -100)
    # keep at least one scored position so the mean is defined
    targets[0, 0] = int(torch.randint(0, v, (1,), generator=g))
    return logits, targets


def test_random_batches_match_cross_entropy_with_ignore_index():
    for seed, (b, t, v) in enumerate([(1, 1, 2), (2, 3, 5), (4, 6, 11), (3, 8, 32)]):
        logits, targets = _random_batch(seed, b, t, v)
        mean_nll, _ = _call(logits, targets)
        want, _ = _oracle(logits, targets)
        if not torch.allclose(mean_nll, want, atol=ATOL, rtol=RTOL):
            raise AssertionError(
                "mean NLL differs from cross-entropy: check the log-sum-exp, the gather of the "
                "target logit, and that the mean divides by the non-ignored count"
            )


def test_perplexity_is_exp_of_mean_nll_and_matches_oracle():
    for seed, (b, t, v) in enumerate([(2, 4, 7), (5, 2, 13)]):
        logits, targets = _random_batch(100 + seed, b, t, v)
        mean_nll, ppl = _call(logits, targets)
        _, want_ppl = _oracle(logits, targets)
        if not torch.allclose(ppl, torch.exp(mean_nll), atol=ATOL, rtol=RTOL):
            raise AssertionError(
                "perplexity must equal exp of the mean NLL in nats, not another base"
            )
        if not torch.allclose(ppl, want_ppl, atol=ATOL, rtol=RTOL):
            raise AssertionError("perplexity differs from exp(cross-entropy) on the same tokens")


def test_denominator_counts_scored_tokens_when_a_whole_row_is_ignored():
    logits = torch.randn((2, 3, 6), generator=_gen(7), dtype=torch.float64)
    targets = torch.tensor([[-100, -100, -100], [2, -100, 4]])  # row 0 is fully ignored
    mean_nll, _ = _call(logits, targets)
    want, _ = _oracle(logits, targets)
    if not torch.allclose(mean_nll, want, atol=ATOL, rtol=RTOL):
        raise AssertionError(
            "a fully ignored row must add nothing: divide by the scored tokens, not by B or B * T"
        )


def test_single_scored_token_equals_its_own_nll():
    logits = torch.tensor([[[0.5, -1.0, 2.0], [9.0, 9.0, 9.0]]], dtype=torch.float64)
    targets = torch.tensor([[2, -100]])
    mean_nll, _ = _call(logits, targets)
    want = math.log(math.exp(0.5) + math.exp(-1.0) + math.exp(2.0)) - 2.0
    if abs(float(mean_nll) - want) > ATOL:
        raise AssertionError(
            "with one scored token the mean must be that token's NLL: logsumexp(z) - z_target"
        )


def test_adding_a_constant_to_each_row_leaves_nll_unchanged():
    logits, targets = _random_batch(21, 3, 4, 9)
    base, _ = _call(logits, targets)
    shifted = logits + torch.arange(3, dtype=torch.float64).reshape(3, 1, 1) * 7.5
    moved, _ = _call(shifted, targets)
    if not torch.allclose(base, moved, atol=ATOL, rtol=RTOL):
        raise AssertionError(
            "softmax is invariant to a per-row constant: the log-sum-exp must subtract the row "
            "maximum, not use a fixed reference"
        )


def test_large_common_offset_stays_finite_and_matches_oracle():
    logits, targets = _random_batch(31, 2, 5, 8)
    offset = logits + 1000.0  # naive exp(1000) overflows float64
    mean_nll, ppl = _call(offset, targets)
    if not bool(torch.isfinite(mean_nll)) or not bool(torch.isfinite(ppl)):
        raise AssertionError("large logits produced inf or NaN: use the stable log-sum-exp")
    want, _ = _oracle(offset, targets)
    if not torch.allclose(mean_nll, want, atol=ATOL, rtol=RTOL):
        raise AssertionError("large-logit NLL differs from the reference: subtract the row maximum")


def test_very_large_logit_gap_matches_closed_form():
    logits = torch.tensor([[[1000.0, 999.0]]], dtype=torch.float64)
    targets = torch.tensor([[1]])
    mean_nll, ppl = _call(logits, targets)
    want = 1.0 + math.log(1.0 + math.exp(-1.0))
    if abs(float(mean_nll) - want) > ATOL or abs(float(ppl) - (1.0 + math.e)) > ATOL:
        raise AssertionError(
            "a gap of 1 between logits 1000 and 999 must give NLL 1 + log(1 + e^-1)"
        )


def test_float32_input_returns_float32_within_declared_bound():
    logits, targets = _random_batch(41, 2, 6, 10, dtype=torch.float32)
    mean_nll, ppl = _call(logits, targets)
    want, want_ppl = _oracle(logits, targets)
    if mean_nll.dtype != torch.float32:
        raise AssertionError("float32 logits must give a float32 mean NLL")
    if not torch.allclose(mean_nll, want, atol=ATOL_F32, rtol=RTOL_F32):
        raise AssertionError(
            "float32 mean NLL differs from the reference beyond the declared bound"
        )
    if not torch.allclose(ppl, want_ppl, atol=ATOL_F32, rtol=RTOL_F32):
        raise AssertionError(
            "float32 perplexity differs from the reference beyond the declared bound"
        )


def test_custom_ignore_index_that_is_a_valid_class_id():
    logits, targets = _random_batch(51, 2, 4, 6, ignore_rate=0.0)
    targets = targets.clone()
    targets[targets == 3] = 1  # no target equals 3 yet, so ignore_index=3 drops nothing else
    targets[0, 1] = 3  # now ignore id 3 marks one position
    mean_nll, _ = _call(logits, targets, ignore_index=3)
    want, _ = _oracle(logits, targets, ignore_index=3)
    if not torch.allclose(mean_nll, want, atol=ATOL, rtol=RTOL):
        raise AssertionError(
            "a custom ignore_index must be skipped even when it is a valid class id"
        )


def test_single_class_vocabulary_has_zero_nll_and_perplexity_one():
    logits = torch.randn((2, 3, 1), generator=_gen(61), dtype=torch.float64)
    targets = torch.zeros((2, 3), dtype=torch.int64)
    mean_nll, ppl = _call(logits, targets)
    if abs(float(mean_nll)) > ATOL or abs(float(ppl) - 1.0) > ATOL:
        raise AssertionError("with one class every NLL is 0 and the perplexity is exactly 1")


def test_targets_out_of_range_raise_value_error():
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    _expect_value_error(
        lambda: masked_nll_and_perplexity(logits, torch.tensor([[0, 4]])),
        "a target equal to V must raise ValueError: valid ids are 0 to V - 1",
    )
    _expect_value_error(
        lambda: masked_nll_and_perplexity(logits, torch.tensor([[0, -3]])),
        "a target below the ignore id must raise ValueError, not index from the end",
    )


def test_all_targets_ignored_raises_value_error():
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    _expect_value_error(
        lambda: masked_nll_and_perplexity(logits, torch.tensor([[-100, -100]])),
        "an all-ignored batch has no defined mean and must raise ValueError",
    )


def test_float_targets_raise_value_error():
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    _expect_value_error(
        lambda: masked_nll_and_perplexity(logits, torch.tensor([[0.0, 1.0]])),
        "targets must be int64; float targets are invalid",
    )


def test_two_dimensional_logits_raise_value_error():
    _expect_value_error(
        lambda: masked_nll_and_perplexity(
            torch.zeros((2, 4), dtype=torch.float64), torch.tensor([0, 1])
        ),
        "logits must be 3-dimensional (B, T, V)",
    )


def test_target_shape_mismatch_raises_value_error():
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    _expect_value_error(
        lambda: masked_nll_and_perplexity(logits, torch.tensor([[0, 1, 2]])),
        "targets must have shape (B, T) matching the logits",
    )


def test_integer_logits_raise_value_error():
    _expect_value_error(
        lambda: masked_nll_and_perplexity(
            torch.zeros((1, 2, 4), dtype=torch.int64), torch.tensor([[0, 1]])
        ),
        "logits must be floating point: integer logits cannot give a log-probability",
    )


def test_bool_ignore_index_raises_value_error():
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    _expect_value_error(
        lambda: masked_nll_and_perplexity(logits, torch.tensor([[0, 1]]), ignore_index=True),
        "a bool ignore_index must raise ValueError",
    )


def test_inputs_are_not_mutated():
    logits, targets = _random_batch(71, 2, 3, 5)
    _call(logits, targets)  # _call checks equality with snapshots and raises if anything changed


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    rows=st.integers(min_value=1, max_value=3),
    steps=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_mean_nll_is_per_token_average_and_perplexity_is_at_least_one(rows, steps, seed):
    vocab = 5
    g = _gen(seed)
    logits = torch.randn((rows, steps, vocab), generator=g, dtype=torch.float64) * 3.0
    targets = torch.randint(0, vocab, (rows, steps), generator=g)
    targets[0, 0] = 2  # guarantee one scored token
    mean_nll, ppl = _call(logits, targets)
    per_token = []
    for n in range(rows):
        for t in range(steps):
            z = logits[n, t]
            m = float(z.max())
            lse = m + math.log(sum(math.exp(float(v) - m) for v in z))
            per_token.append(lse - float(z[int(targets[n, t])]))
    want = sum(per_token) / len(per_token)
    if abs(float(mean_nll) - want) > 1e-9:
        raise AssertionError("mean NLL must average the per-token NLLs over all scored tokens")
    if float(ppl) < 1.0 - 1e-12:
        raise AssertionError("perplexity cannot be below 1 for a proper log-probability")


def test_row_with_a_large_logit_gap_gives_a_finite_nll():
    logits = torch.tensor([[[0.0, 1000.0]]], dtype=torch.float64)
    targets = torch.tensor([[0]])
    mean_nll, _ = _call(logits, targets)
    value = float(mean_nll)
    if not math.isfinite(value) or abs(value - 1000.0) > ATOL:
        raise AssertionError(
            "large logits must keep the NLL finite and exact: the stable log-sum-exp needs the "
            "row maximum, not the minimum"
        )


def test_ignore_index_outside_the_vocabulary_is_still_skipped():
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    for ignore in (4, 100):
        targets = torch.tensor([[1, ignore]])
        try:
            mean_nll, _ = masked_nll_and_perplexity(logits, targets, ignore_index=ignore)
        except Exception as exc:
            raise AssertionError(
                f"an ignore_index that is not a valid class id must still be skipped, not "
                f"indexed ({type(exc).__name__})"
            ) from None
        if abs(float(mean_nll) - math.log(4)) > ATOL:
            raise AssertionError(
                "ignored positions must add nothing even when their id lies outside the vocabulary"
            )
