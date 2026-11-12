# Hidden tests: oracle comparison with PyTorch's cross-entropy, shift and mask checks, stability,
# dtype, a finite-difference gradient check and the input-validation rules.
# The oracle is used only here; the submission may not use torch.nn or its functional module.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import sequence_log_probs

from dogma_data import rng as data_rng
from dogma_data import sample_rows

ATOL = 1e-9
RTOL = 1e-9
# float32 keeps about 7 significant digits; the sums here add up to 15 terms, so 1e-5 leaves room
# for rounding without hiding a wrong mask or a shift error, which is off by whole nats.
FLOAT32_ATOL = 1e-5


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
    key = f"p01-sequence-log-probs-with-response-mask:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _logits(shape, seed, scale=2.0, offset=0.0, dtype=torch.float64):
    return (_real_values(_gen(seed), [shape], dtype) * scale + offset).clone()


def _ids(shape, vocab, seed):
    return torch.randint(0, vocab, shape, generator=_gen(seed), dtype=torch.int64)


def _mask(shape, seed):
    """Random response mask with the first position always False (nothing predicts it)."""
    m = torch.special.ndtr(_real_values(_gen(seed), [shape])) < 0.6
    m[:, 0] = False
    return m


def _reference(logits, input_ids, mask):
    """Independent oracle: cross-entropy of each actual next token, summed over response tokens."""
    b, t, v = logits.shape
    if t == 1:
        return torch.zeros(b, dtype=logits.dtype)
    ce = ref_f.cross_entropy(
        logits[:, :-1, :].reshape(-1, v),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(b, t - 1)
    return -(ce * mask[:, 1:].to(logits.dtype)).sum(dim=1)


def _call(logits, input_ids, mask):
    snap = (logits.clone(), input_ids.clone(), mask.clone())
    try:
        out = sequence_log_probs(logits, input_ids, mask)
    except ValueError as exc:
        pytest.fail(
            f"valid input raised ValueError ({exc}): check the contract's shape and dtype rules",
            pytrace=False,
        )
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, indexing and dtypes",
            pytrace=False,
        )
    if not torch.is_tensor(out):
        pytest.fail(
            "return a torch tensor of shape (B,), not a Python number or list", pytrace=False
        )
    if tuple(out.shape) != (logits.shape[0],):
        pytest.fail(
            "output must have shape (B,): one summed log-probability per row", pytrace=False
        )
    if out.dtype != logits.dtype:
        pytest.fail("output must keep the dtype of logits; do not downcast", pytrace=False)
    if not (
        torch.equal(snap[0], logits)
        and torch.equal(snap[1], input_ids)
        and torch.equal(snap[2], mask)
    ):
        pytest.fail("an input tensor was modified in place; build new tensors", pytrace=False)
    return out


def _close(actual, expected, message, atol=ATOL, rtol=RTOL):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        pytest.fail(message, pytrace=False)


def _expect_value_error(thunk, message):
    try:
        thunk()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail(message, pytrace=False)


def test_sums_match_reference_on_random_batches():
    for seed, (b, t, v) in enumerate([(1, 2, 3), (3, 5, 7), (4, 8, 11), (2, 16, 5)]):
        logits = _logits((b, t, v), seed)
        ids = _ids((b, t), v, seed + 10)
        mask = _mask((b, t), seed + 20)
        out = _call(logits, ids, mask)
        _close(
            out,
            _reference(logits, ids, mask),
            "sequence log-probs disagree with the reference: check the one-token shift "
            "(position t is scored by logits at t - 1), the mask alignment, and that the sum "
            "runs over response tokens only",
        )


def test_next_token_is_scored_by_previous_position_not_the_same_position():
    logits = _logits((2, 6, 9), 77)
    ids = _ids((2, 6), 9, 78)
    mask = torch.ones((2, 6), dtype=torch.bool)
    mask[:, 0] = False
    out = _call(logits, ids, mask)
    unshifted = torch.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    unshifted_sum = unshifted[:, 1:].sum(dim=1)
    if torch.allclose(out, unshifted_sum, atol=1e-9):
        pytest.fail(
            "scored tokens with the logits at their own position: the next-token shift is missing",
            pytrace=False,
        )
    _close(
        out, _reference(logits, ids, mask), "shift mistake: logits must score the following token"
    )


def test_mask_on_a_token_is_not_offset_by_one_position():
    logits = _logits((1, 4, 6), 90)
    ids = _ids((1, 4), 6, 91)
    mask = torch.tensor([[False, True, False, False]])
    out = _call(logits, ids, mask)
    want = _reference(logits, ids, mask)
    _close(
        out,
        want,
        "mask is misaligned with the scored tokens: the mask at position t selects token t",
    )


def test_row_without_response_tokens_sums_to_zero_and_batch_of_none_gives_zeros():
    logits = _logits((3, 4, 5), 100)
    ids = _ids((3, 4), 5, 101)
    mask = torch.zeros((3, 4), dtype=torch.bool)
    out = _call(logits, ids, mask)
    _close(
        out, torch.zeros(3, dtype=torch.float64), "a row with no response tokens must sum to zero"
    )


def test_single_position_sequence_has_nothing_to_score():
    logits = _logits((2, 1, 4), 110)
    ids = _ids((2, 1), 4, 111)
    mask = torch.zeros((2, 1), dtype=torch.bool)
    out = _call(logits, ids, mask)
    _close(
        out,
        torch.zeros(2, dtype=torch.float64),
        "a single-position sequence has no scored token: expected zeros",
    )


def test_large_logit_magnitudes_stay_finite_and_match_reference():
    logits = _logits((2, 5, 6), 120, scale=1.0, offset=1000.0)
    ids = _ids((2, 5), 6, 121)
    mask = _mask((2, 5), 122)
    out = _call(logits, ids, mask)
    assert torch.isfinite(out).all(), (
        "log-probs overflowed with large logits: use log_softmax, not exp then log"
    )
    _close(
        out,
        _reference(logits, ids, mask),
        "large-logit log-probs disagree with the reference: subtract the max "
        "inside the log-softmax",
    )


def test_very_unlikely_tokens_do_not_underflow_to_negative_infinity():
    logits = torch.zeros((1, 3, 4), dtype=torch.float64)
    # Position 1 scores the token at position 2, which is id 1; make that token extremely unlikely.
    logits[0, 1, 1] = -800.0
    ids = torch.tensor([[0, 2, 1]])
    mask = torch.tensor([[False, True, True]])
    out = _call(logits, ids, mask)
    assert torch.isfinite(out).all(), (
        "a very unlikely token gave -inf: softmax then log underflows; use log_softmax"
    )
    _close(
        out,
        _reference(logits, ids, mask),
        "very unlikely token log-prob disagrees with the reference",
    )


def test_float32_inputs_keep_dtype_and_match_reference():
    logits = _logits((3, 6, 7), 130, dtype=torch.float32)
    ids = _ids((3, 6), 7, 131)
    mask = _mask((3, 6), 132)
    out = _call(logits, ids, mask)
    assert out.dtype == torch.float32, "output must stay float32 when logits are float32"
    want = _reference(logits, ids, mask)
    _close(
        out,
        want,
        "float32 log-probs disagree with the reference beyond rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )


def test_gradient_through_logits_matches_finite_differences_in_float64():
    logits = _logits((2, 4, 5), 140, scale=1.0)
    ids = _ids((2, 4), 5, 141)
    mask = _mask((2, 4), 142)
    x = logits.clone().requires_grad_(True)
    sequence_log_probs(x, ids, mask).sum().backward()
    assert x.grad is not None, (
        "no gradient reached the logits: the output is detached from autograd"
    )

    def total(z):
        return float(sequence_log_probs(z, ids, mask).sum())

    h = 1e-6
    numeric = torch.zeros_like(logits)
    flat = logits.reshape(-1)
    for i in range(flat.numel()):
        up, down = logits.clone(), logits.clone()
        up.view(-1)[i] += h
        down.view(-1)[i] -= h
        numeric.view(-1)[i] = (total(up) - total(down)) / (2 * h)
    _close(
        x.grad,
        numeric,
        "gradient through the logits disagrees with finite differences: "
        "check the log-softmax and the mask",
        atol=1e-7,
        rtol=1e-7,
    )


def test_mask_on_the_first_token_raises_value_error():
    logits = _logits((1, 3, 4), 150)
    ids = _ids((1, 3), 4, 151)
    mask = torch.tensor([[True, True, False]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "a mask on the first token was accepted: the first token has no prediction, "
        "so it must raise ValueError",
    )


def test_token_id_outside_vocabulary_raises_value_error():
    logits = _logits((1, 3, 4), 152)
    ids = torch.tensor([[0, 4, 1]])
    mask = torch.tensor([[False, True, True]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "a token id outside [0, V) was accepted: check the vocabulary bound",
    )


def test_integer_mask_raises_value_error():
    logits = _logits((1, 3, 4), 153)
    ids = _ids((1, 3), 4, 154)
    mask = torch.tensor([[0, 1, 1]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "an integer response mask was accepted: the mask must be boolean so True means response",
    )


def test_mismatched_ids_shape_raises_value_error():
    logits = _logits((2, 3, 4), 155)
    ids = _ids((2, 4), 4, 156)
    mask = torch.ones((2, 3), dtype=torch.bool)
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "input_ids with a shape different from the logits was accepted: check the (B, T) shape",
    )


def test_non_floating_logits_raise_value_error():
    logits = torch.zeros((1, 3, 4), dtype=torch.int64)
    ids = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[False, True, True]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "integer logits were accepted: logits must be floating-point scores",
    )


def test_two_dimensional_logits_raise_value_error():
    logits = _logits((3, 4), 157)
    ids = _ids((3,), 4, 158)
    mask = torch.ones((3,), dtype=torch.bool)
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "2-D logits were accepted: pass (B, T, V) logits",
    )


def test_empty_batch_raises_value_error():
    logits = torch.zeros((0, 3, 4), dtype=torch.float64)
    ids = torch.zeros((0, 3), dtype=torch.int64)
    mask = torch.zeros((0, 3), dtype=torch.bool)
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "an empty batch was accepted: a batch needs at least one row",
    )


def test_response_mask_with_a_shape_other_than_b_t_raises_value_error():
    logits = _logits((1, 3, 4), 160)
    ids = _ids((1, 3), 4, 161)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "a response mask whose shape differs from (B, T) was accepted: it must match the logits",
    )


def test_input_ids_with_a_shape_other_than_b_t_raises_value_error():
    # The mask starts with False, so only the input_ids shape can be the mistake here.
    logits = _logits((2, 3, 4), 162)
    ids = _ids((2, 4), 4, 163)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "input_ids with a shape other than (B, T) of the logits was accepted: check its shape",
    )


def test_float_token_ids_raise_value_error():
    logits = _logits((1, 3, 4), 164)
    ids = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.tensor([[False, True, True]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "float token ids were accepted: input_ids must be an integer tensor",
    )


def test_bool_token_ids_raise_value_error():
    logits = _logits((1, 3, 4), 165)
    ids = torch.tensor([[True, False, True]])
    mask = torch.tensor([[False, True, True]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "boolean token ids were accepted: input_ids must be integer ids, not booleans",
    )


def test_first_token_marked_in_any_row_raises_value_error():
    # Row 0 is fine; row 1 marks its first token, which has no prediction.
    logits = _logits((2, 3, 4), 166)
    ids = _ids((2, 3), 4, 167)
    mask = torch.tensor([[False, True, False], [True, True, True]])
    _expect_value_error(
        lambda: sequence_log_probs(logits, ids, mask),
        "a first token marked in any row must raise ValueError, not only when every row marks it",
    )


def test_single_entry_vocabulary_is_a_valid_input():
    # V >= 1 is allowed. With one vocabulary entry every log-probability is log(1) = 0.
    logits = torch.zeros((1, 3, 1), dtype=torch.float64)
    ids = torch.zeros((1, 3), dtype=torch.int64)
    mask = torch.tensor([[False, True, True]])
    out = _call(logits, ids, mask)
    _close(
        out,
        torch.zeros(1, dtype=torch.float64),
        "a one-entry vocabulary is valid and every token has probability 1, so the sum is zero",
    )


def test_float16_logits_are_valid_floating_point_input():
    logits = _logits((2, 4, 5), 911, scale=1.0).to(torch.float16)
    ids = _ids((2, 4), 5, 912)
    mask = _mask((2, 4), 913)
    out = _call(logits, ids, mask)
    assert out.dtype == torch.float16, "float16 logits are floating-point: the result keeps float16"
    want = _reference(logits.to(torch.float64), ids, mask)
    _close(
        out.to(torch.float64),
        want,
        "float16 logits are floating-point and must be scored, not refused",
        atol=2e-2,
        rtol=2e-2,
    )


def test_int32_token_ids_are_valid_integer_input():
    logits = _logits((2, 4, 6), 921)
    ids = _ids((2, 4), 6, 922).to(torch.int32)
    mask = _mask((2, 4), 923)
    out = _call(logits, ids, mask)
    want = _reference(logits, ids.to(torch.int64), mask)
    _close(
        out,
        want,
        "int32 token ids are integer ids and must be scored like int64 ids",
        atol=ATOL,
        rtol=RTOL,
    )
