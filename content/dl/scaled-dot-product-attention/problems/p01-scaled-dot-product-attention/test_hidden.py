# Hidden tests: oracle comparisons with torch.nn.functional, scale and stability checks,
# degenerate and invalid inputs, dtype, and mutation. Messages name the kind of mistake.
import functools
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - oracle only
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import scaled_dot_product_attention

from dogma_data import rng as data_rng
from dogma_data import sample_rows

DT = torch.float64


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
    key = f"p01-scaled-dot-product-attention:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def checked(call):  # the learner gets clones in oracle tests; see test_inputs_are_not_mutated
    try:
        out = call()
    except Exception:
        pytest.fail(
            "valid input raised an exception: honor the documented interface", pytrace=False
        )
    if not (isinstance(out, tuple) and len(out) == 2):
        pytest.fail("return a tuple (output, weights)", pytrace=False)
    return out


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail("invalid input must raise ValueError but returned a result", pytrace=False)


def oracle_weights(q, k, mask=None):
    """Independent reference: float64 softmax with -inf masking; fully masked rows set to zero."""
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~torch.broadcast_to(mask, scores.shape), float("-inf"))
    w = F.softmax(scores, dim=-1)
    return torch.nan_to_num(w, nan=0.0)


def has_allowed(scores_shape, mask):
    if mask is None:
        return torch.ones(scores_shape[:-1], dtype=torch.bool)
    return torch.broadcast_to(mask, scores_shape).any(dim=-1)


def close(actual, expected, message):
    if tuple(actual.shape) != tuple(expected.shape):
        pytest.fail(f"output shape differs from contract: {message}", pytrace=False)
    if not torch.allclose(actual, expected, atol=1e-9, rtol=1e-9):
        pytest.fail(message, pytrace=False)


def random_inputs(seed, batch, L, S, d_k, d_v, scale=1.0):  # noqa: PLR0917 - shape test helper
    g = torch.Generator().manual_seed(seed)
    q = scale * _real_values(g, [*batch, L, d_k], DT)
    k = scale * _real_values(g, [*batch, S, d_k], DT)
    v = _real_values(g, [*batch, S, d_v], DT)
    return q, k, v


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 4), st.integers(1, 5), st.integers(1, 6), st.integers(0, 10_000))
def test_random_unmasked_output_and_weights_match_oracle(L, S, d_k, seed):
    q, k, v = random_inputs(seed, (2,), L, S, d_k, 3)
    out, w = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone())
    )
    expected_w = oracle_weights(q, k)
    close(w, expected_w, "weights differ from the softmax oracle: check scale and key-axis softmax")
    close(out, expected_w @ v, "output differs from weights @ v")


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 4), st.integers(1, 5), st.integers(0, 10_000))
def test_random_masked_output_matches_functional_oracle(L, S, seed):
    g = torch.Generator().manual_seed(seed + 1)
    q, k, v = random_inputs(seed, (3,), L, S, 4, 2)
    mask = torch.special.ndtr(_real_values(g, [3, L, S])) > 0.4
    out, w = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone(), mask)
    )
    expected_w = oracle_weights(q, k, mask)
    close(
        w,
        expected_w,
        "masked weights differ from the oracle: check mask polarity (True = may attend)",
    )
    rows = has_allowed(w.shape, mask)
    ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    close(out[rows], ref_out[rows], "masked output differs from the functional oracle on live rows")


def test_weight_rows_sum_to_one_and_masked_pairs_are_zero():
    q, k, v = random_inputs(31, (2,), 4, 5, 3, 2)
    mask = torch.tensor([[True, False, True, True, False]] * 4)
    _, w = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone(), mask)
    )
    if not torch.allclose(w.sum(dim=-1), torch.ones(2, 4, dtype=DT), atol=1e-9):
        pytest.fail(
            "weight rows must sum to 1 over the keys: the softmax axis is wrong", pytrace=False
        )
    if torch.count_nonzero(w * (~mask).to(DT)) != 0:
        pytest.fail("masked pairs must receive exactly zero weight", pytrace=False)


def test_scale_is_square_root_of_key_width_at_d_k_64():
    # At d_k = 64 a missing or wrong scale changes the softmax sharply; compare with the oracle.
    q, k, v = random_inputs(41, (), 3, 6, 64, 2)
    _, w = checked(functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone()))
    close(
        w,
        oracle_weights(q, k),
        "weights differ at d_k=64: divide scores by sqrt(d_k), not by d_k or not at all",
    )


def test_large_logits_stay_finite_and_match_oracle():
    q, k, v = random_inputs(51, (2,), 3, 4, 4, 2, scale=1e3)
    out, w = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone())
    )
    if not (torch.isfinite(out).all() and torch.isfinite(w).all()):
        pytest.fail("large logits overflowed: subtract the row maximum before exp", pytrace=False)
    close(w, oracle_weights(q, k), "large-logit weights differ from the oracle (stability)")


def test_large_logits_with_mask_stay_finite():
    q, k, v = random_inputs(52, (), 4, 4, 3, 2, scale=1e3)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, False],
            [True, False, True, True],
            [False, True, False, True],
        ]
    )
    out, w = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone(), mask)
    )
    if not (torch.isfinite(out).all() and torch.isfinite(w).all()):
        pytest.fail(
            "masked large logits produced NaN or inf: guard the max-subtraction", pytrace=False
        )
    close(w, oracle_weights(q, k, mask), "masked large-logit weights differ from the oracle")


def test_fully_masked_rows_are_zero_and_other_rows_unchanged():
    q, k, v = random_inputs(61, (2,), 3, 4, 2, 2)
    mask = torch.tensor(
        [[True, True, False, False], [False, False, False, False], [False, True, True, False]]
    ).expand(2, 3, 4)
    out, w = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone(), mask)
    )
    if not (torch.isfinite(out).all() and torch.isfinite(w).all()):
        pytest.fail(
            "fully masked row produced NaN or inf: naive max-subtraction gives -inf minus -inf",
            pytrace=False,
        )
    if torch.count_nonzero(w[:, 1]) != 0 or torch.count_nonzero(out[:, 1]) != 0:
        pytest.fail("fully masked row must have zero weights and zero output", pytrace=False)
    live = torch.tensor([True, False, True]).expand(2, 3)
    close(
        w[live],
        oracle_weights(q, k, mask)[live],
        "live rows changed when another row is fully masked",
    )


def test_boolean_mask_broadcasts_over_batch_and_rejects_bad_shape():
    q, k, v = random_inputs(71, (2, 3), 2, 4, 3, 2)
    shared = torch.tensor([[True, False, True, True], [True, True, False, False]])
    full = shared.expand(2, 3, 2, 4)
    out_a, _ = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone(), shared)
    )
    out_b, _ = checked(
        functools.partial(scaled_dot_product_attention, q.clone(), k.clone(), v.clone(), full)
    )
    close(out_a, out_b, "a (L, S) mask must broadcast over the batch axes")
    rejects(lambda: scaled_dot_product_attention(q, k, v, torch.ones(4, 2, dtype=torch.bool)))


def test_additive_float_mask_is_rejected():
    q, k, v = random_inputs(72, (), 2, 3, 2, 2)
    rejects(lambda: scaled_dot_product_attention(q, k, v, torch.zeros(2, 3)))


def test_zero_size_and_low_rank_inputs_rejected():
    rejects(
        lambda: scaled_dot_product_attention(torch.ones(0, 2), torch.ones(3, 2), torch.ones(3, 2))
    )
    rejects(
        lambda: scaled_dot_product_attention(torch.ones(2, 2), torch.ones(0, 2), torch.ones(0, 2))
    )
    rejects(lambda: scaled_dot_product_attention(torch.ones(2), torch.ones(3, 2), torch.ones(3, 2)))


def test_mismatched_batch_or_key_width_rejected():
    rejects(
        lambda: scaled_dot_product_attention(
            torch.ones(2, 2, 3), torch.ones(3, 3, 3), torch.ones(3, 3, 3)
        )
    )
    rejects(
        lambda: scaled_dot_product_attention(torch.ones(2, 3), torch.ones(4, 2), torch.ones(4, 2))
    )


def test_mixed_dtype_rejected_and_float32_stays_float32():
    q, k, v = random_inputs(81, (), 2, 3, 2, 2)
    rejects(lambda: scaled_dot_product_attention(q, k.float(), v))
    out, w = checked(
        functools.partial(scaled_dot_product_attention, q.float(), k.float(), v.float())
    )
    if out.dtype != torch.float32 or w.dtype != torch.float32:
        pytest.fail("output and weights must keep the input dtype", pytrace=False)


def test_inputs_are_not_mutated():
    q, k, v = random_inputs(91, (2,), 3, 4, 3, 2)
    mask = torch.special.ndtr(_real_values(torch.Generator().manual_seed(257), [2, 3, 4])) > 0.3
    originals = [t.clone() for t in (q, k, v, mask)]
    checked(functools.partial(scaled_dot_product_attention, q, k, v, mask))
    for before, after in zip(originals, (q, k, v, mask), strict=True):
        if not torch.equal(before, after):
            pytest.fail(
                "inputs must not be mutated (for example, scaling q in place)", pytrace=False
            )


def _rejects_with(message, call):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{message}: raised {type(exc).__name__} instead of ValueError", pytrace=False)
    pytest.fail(f"{message}: no ValueError was raised", pytrace=False)


def test_all_integer_tensors_are_rejected_with_value_error():
    q = torch.ones(2, 3, dtype=torch.int64)
    k = torch.ones(4, 3, dtype=torch.int64)
    v = torch.ones(4, 2, dtype=torch.int64)
    _rejects_with(
        "integer q, k and v must be rejected with ValueError: attention needs "
        "floating-point tensors",
        lambda: scaled_dot_product_attention(q, k, v),
    )
