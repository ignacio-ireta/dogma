# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show hidden
# inputs, shapes or expected values. The oracle uses complex arithmetic with torch.polar, an
# independent route to the same rotation, which the learner may not import.
from __future__ import annotations

import copy
import math

import pytest
import torch
from submission import apply_rope_qk

from dogma_data import rng as data_rng
from dogma_data import sample_rows

ATOL = 1e-10
RTOL = 1e-10


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
    key = f"p01-half-split-rope-with-offset:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def _angles(tokens, d, offset, dtype=torch.float64):
    """Angle table phi[t, i] = (offset + t) * 10000**(-2 i / d), shape (T, d/2), float64."""
    t = torch.arange(offset, offset + tokens, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(d // 2, dtype=torch.float64).unsqueeze(0)
    return (t * torch.pow(torch.tensor(10000.0, dtype=torch.float64), -2.0 * i / d)).to(dtype)


def _ref_half_split(x, offset):
    """Complex reference: z_i = x_i + i x_{i+d/2}, multiplied by exp(i phi), split back."""
    d = x.shape[-1]
    half = d // 2
    x64 = x.double()
    z = torch.complex(x64[..., :half], x64[..., half:])
    rot = z * torch.polar(
        torch.ones_like(_angles(x.shape[-2], d, offset)), _angles(x.shape[-2], d, offset)
    )
    return torch.cat([rot.real, rot.imag], dim=-1).to(x.dtype)


def _ref_interleaved(x, offset):
    """Complex reference for the adjacent-pair layout: z_i = x_{2i} + i x_{2i+1}."""
    d = x.shape[-1]
    x64 = x.double()
    z = torch.complex(x64[..., 0::2], x64[..., 1::2])
    rot = z * torch.polar(
        torch.ones_like(_angles(x.shape[-2], d, offset)), _angles(x.shape[-2], d, offset)
    )
    out = torch.empty_like(x64)
    out[..., 0::2] = rot.real
    out[..., 1::2] = rot.imag
    return out.to(x.dtype)


def _run(q, k, offset, *, pass_through=False):
    """Call the learner on copies, so an in-place write cannot corrupt the oracle's inputs."""
    call_q, call_k = (q, k) if pass_through else (q.clone(), k.clone())
    try:
        out = apply_rope_qk(call_q, call_k, offset)
    except ValueError:
        pytest.fail("valid input was rejected: check the contract")
    except Exception as exc:  # any crash on valid input is a contract failure
        pytest.fail(
            f"valid input crashed ({type(exc).__name__}): check the (B, H, T, d) indexing and that "
            "the angles broadcast over the batch and head axes"
        )
    if not isinstance(out, tuple) or len(out) != 2:
        pytest.fail("return exactly two tensors, (q_rot, k_rot): the value tensor is never rotated")
    for name, tensor, source in (("q_rot", out[0], q), ("k_rot", out[1], k)):
        if not torch.is_tensor(tensor):
            pytest.fail(f"{name} must be a tensor")
        if tuple(tensor.shape) != tuple(source.shape):
            pytest.fail(f"{name} must keep the input shape (B, H, T, d)")
        if tensor.dtype != source.dtype:
            pytest.fail(f"{name} must keep the input dtype")
        if not torch.isfinite(tensor).all():
            pytest.fail(f"{name} is not finite: check the angle table for large offsets")
    return out


def _assert_close(actual, expected, what, hint, tol=ATOL):
    if not torch.allclose(actual, expected, atol=tol, rtol=tol):
        pytest.fail(f"{what} differ from the complex-number reference: {hint}")


QUERY_HINT = "check the pairing (i with i + d/2), theta_i = 10000 ** (-2 i / d) and the sine sign"
KEY_HINT = "keys need the same offset and frequencies as queries; values are not rotated"


@pytest.mark.parametrize(
    "shape, offset",
    [((1, 1, 1, 2), 0), ((2, 3, 4, 8), 0), ((2, 2, 5, 6), 3), ((1, 4, 7, 16), 11),
     ((3, 1, 2, 4), 100)],
    ids=["single-token-two-wide", "batched-zero-offset", "offset-three",
         "eight-wide-offset-eleven", "batched-large-offset"],
)  # fmt: skip
def test_matches_complex_reference_on_seeded_tensors(shape, offset):
    gen = torch.Generator().manual_seed(sum(shape) + offset)
    q = _real_values(gen, [*shape], torch.float64)
    k = _real_values(gen, [*shape], torch.float64)
    q_rot, k_rot = _run(q, k, offset)
    _assert_close(q_rot, _ref_half_split(q, offset), "rotated queries", QUERY_HINT)
    _assert_close(k_rot, _ref_half_split(k, offset), "rotated keys", KEY_HINT)


def test_half_split_is_not_the_adjacent_pair_layout():
    gen = torch.Generator().manual_seed(2)
    q = _real_values(gen, [2, 2, 3, 6], torch.float64)
    k = _real_values(gen, [2, 2, 3, 6], torch.float64)
    # perm puts the even features first, then the odd ones: then pair i of the half-split layout
    # is exactly the adjacent pair (2i, 2i+1) of the original tensor.
    perm = torch.cat([torch.arange(0, 6, 2), torch.arange(1, 6, 2)])
    q_rot, _ = _run(q[..., perm], k[..., perm], 4)
    interleaved = _ref_interleaved(q, 4)
    if not torch.allclose(q_rot, interleaved[..., perm], atol=ATOL, rtol=RTOL):
        pytest.fail(
            "pairs are in the adjacent layout: half-split pairs feature i with feature i + d/2"
        )


def test_query_key_scores_depend_only_on_relative_offset():
    gen = torch.Generator().manual_seed(5)
    q = _real_values(gen, [1, 2, 6, 8], torch.float64)
    k = _real_values(gen, [1, 2, 6, 8], torch.float64)
    q0, k0 = _run(q, k, 0)
    q7, k7 = _run(q, k, 7)
    s0 = q0 @ k0.transpose(-1, -2)
    s7 = q7 @ k7.transpose(-1, -2)
    if not torch.allclose(s0, s7, atol=1e-9, rtol=1e-9):
        pytest.fail(
            "query-key scores change under a common shift: the offset must move q and k alike"
        )


def test_offset_applied_to_only_one_side_is_detected():
    gen = torch.Generator().manual_seed(6)
    q = _real_values(gen, [1, 1, 4, 4], torch.float64)
    k = _real_values(gen, [1, 1, 4, 4], torch.float64)
    q_rot, k_rot = _run(q, k, 5)
    k_wrong = _run(q, k, 0)[1]  # keys at position 0 while queries sit at offset 5
    if torch.allclose(q_rot @ k_rot.transpose(-1, -2), q_rot @ k_wrong.transpose(-1, -2)):
        pytest.fail("keys are rotated at the wrong position: use the same offset for q and k")


def test_cached_decoding_new_token_matches_full_sequence():
    gen = torch.Generator().manual_seed(7)
    full_q = _real_values(gen, [2, 2, 5, 8], torch.float64)
    full_k = _real_values(gen, [2, 2, 5, 8], torch.float64)
    q_all, k_all = _run(full_q, full_k, 0)
    past = 4
    # The last token alone, with positions starting at past_len: this is one decoding step.
    q_last, k_last = _run(full_q[..., past:, :], full_k[..., past:, :], past)
    if not torch.allclose(q_last, q_all[..., past:, :], atol=ATOL, rtol=RTOL):
        pytest.fail(
            "cached decoding disagrees with the full pass: positions must start at past_len"
        )
    if not torch.allclose(k_last, k_all[..., past:, :], atol=ATOL, rtol=RTOL):
        pytest.fail(
            "cached decoding disagrees for keys: positions must start at past_len for keys too"
        )


def test_position_zero_is_identity_for_every_row_when_offset_is_zero():
    gen = torch.Generator().manual_seed(8)
    x = _real_values(gen, [1, 1, 1, 6], torch.float64)
    q_rot, k_rot = _run(x, x, 0)
    if not torch.allclose(q_rot, x, atol=ATOL) or not torch.allclose(k_rot, x, atol=ATOL):
        pytest.fail("position 0 must be the identity: check the angle at t = 0 and the offset")


def test_rows_keep_their_euclidean_norm():
    gen = torch.Generator().manual_seed(9)
    q = _real_values(gen, [2, 3, 5, 8], torch.float64)
    q_rot, _ = _run(q, q, 9)
    if not torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=ATOL, rtol=RTOL):
        pytest.fail("a rotation must preserve each row's norm: check cos^2 + sin^2 in each pair")


def test_leading_batch_and_head_axes_are_independent():
    gen = torch.Generator().manual_seed(10)
    q = _real_values(gen, [2, 3, 4, 6], torch.float64)
    together, _ = _run(q, q, 2)
    for b in range(2):
        for h in range(3):
            alone, _ = _run(q[b : b + 1, h : h + 1], q[b : b + 1, h : h + 1], 2)
            if not torch.allclose(together[b : b + 1, h : h + 1], alone, atol=ATOL, rtol=RTOL):
                pytest.fail("batch or head rows interact: the angle table must not mix them")


def test_empty_sequence_returns_empty_tensors():
    q = torch.zeros(2, 2, 0, 4, dtype=torch.float64)
    q_rot, k_rot = _run(q, q, 3)
    if q_rot.shape != q.shape or k_rot.shape != q.shape:
        pytest.fail("an empty sequence must return empty tensors of the same width")


@pytest.mark.parametrize("offset", [10**6, 123456], ids=["one-million", "six-digit"])
def test_large_offsets_match_the_reference(offset):
    gen = torch.Generator().manual_seed(offset % 97)
    q = _real_values(gen, [1, 2, 3, 8], torch.float64)
    k = _real_values(gen, [1, 2, 3, 8], torch.float64)
    q_rot, k_rot = _run(q, k, offset)
    # Angles near 1e6 radians carry float64 round-off of about 1e-10, so atol is 1e-8 here.
    _assert_close(q_rot, _ref_half_split(q, offset), "rotated queries", QUERY_HINT, 1e-8)
    _assert_close(k_rot, _ref_half_split(k, offset), "rotated keys", KEY_HINT, 1e-8)


@pytest.mark.parametrize("scale", [1e6, 1e-6], ids=["large-magnitude", "small-magnitude"])
def test_extreme_input_scales_match_the_reference(scale):
    gen = torch.Generator().manual_seed(12)
    q = scale * _real_values(gen, [1, 1, 4, 6], torch.float64)
    k = scale * _real_values(gen, [1, 1, 4, 6], torch.float64)
    q_rot, k_rot = _run(q, k, 1)
    _assert_close(q_rot, _ref_half_split(q, 1), "rotated queries", QUERY_HINT)
    _assert_close(k_rot, _ref_half_split(k, 1), "rotated keys", KEY_HINT)


def test_float32_inputs_return_float32_close_to_reference():
    gen = torch.Generator().manual_seed(14)
    q64 = _real_values(gen, [2, 2, 4, 8], torch.float64)
    q_rot, k_rot = _run(q64.float(), q64.float(), 6)
    if q_rot.dtype != torch.float32 or k_rot.dtype != torch.float32:
        pytest.fail("float32 inputs must return float32 tensors")
    if not torch.allclose(q_rot.double(), _ref_half_split(q64, 6), atol=1e-5, rtol=1e-5):
        pytest.fail("float32 result drifts from the reference beyond single-precision round-off")


def test_inputs_are_not_mutated():
    gen = torch.Generator().manual_seed(15)
    q = _real_values(gen, [1, 2, 3, 4], torch.float64)
    k = _real_values(gen, [1, 2, 3, 4], torch.float64)
    q_before, k_before = copy.deepcopy(q), copy.deepcopy(k)
    _run(q, k, 2, pass_through=True)
    if not torch.equal(q, q_before) or not torch.equal(k, k_before):
        pytest.fail("the input tensors were modified in place: write into new tensors")


@pytest.mark.parametrize(
    "case",
    [
        "odd-width",
        "zero-width",
        "three-dimensional",
        "shape-mismatch",
        "dtype-mismatch",
        "integer-input",
        "negative-offset",
        "float-offset",
        "bool-offset",
        "not-a-tensor",
    ],
    ids=[
        "odd-width",
        "zero-width",
        "three-dimensional",
        "shape-mismatch",
        "dtype-mismatch",
        "integer-input",
        "negative-offset",
        "float-offset",
        "bool-offset",
        "not-a-tensor",
    ],
)
def test_invalid_inputs_raise_value_error(case):
    q = torch.ones(1, 2, 3, 4, dtype=torch.float64)
    k, offset = q.clone(), 2
    if case == "odd-width":
        q = k = torch.ones(1, 2, 3, 5, dtype=torch.float64)
    elif case == "zero-width":
        q = k = torch.ones(1, 2, 3, 0, dtype=torch.float64)
    elif case == "three-dimensional":
        q = k = torch.ones(2, 3, 4, dtype=torch.float64)
    elif case == "shape-mismatch":
        k = torch.ones(1, 2, 4, 4, dtype=torch.float64)
    elif case == "dtype-mismatch":
        k = k.float()
    elif case == "integer-input":
        q = k = torch.ones(1, 2, 3, 4, dtype=torch.int64)
    elif case == "negative-offset":
        offset = -1
    elif case == "float-offset":
        offset = 2.0
    elif case == "bool-offset":
        offset = True
    elif case == "not-a-tensor":
        q = q.tolist()
    try:
        apply_rope_qk(q, k, offset)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input ({case}) raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"invalid input ({case}) was accepted instead of raising ValueError")


def test_offsets_beyond_float32_precision_keep_exact_positions():
    # 2**24 + 1 is not representable in float32, so a position computed in float32 is off by one.
    offset = 2**24 + 1
    gen = torch.Generator().manual_seed(41)
    q = _real_values(gen, [1, 1, 2, 4], torch.float64)
    k = _real_values(gen, [1, 1, 2, 4], torch.float64)
    q_rot, k_rot = _run(q, k, offset)
    # The tolerance is loose on purpose: one wrong position moves the angle by about one radian.
    _assert_close(q_rot, _ref_half_split(q, offset), "rotated queries", QUERY_HINT, 1e-6)
    _assert_close(k_rot, _ref_half_split(k, offset), "rotated keys", KEY_HINT, 1e-6)


def test_empty_batch_returns_empty_tensors():
    # B = 0 is a valid batch size: no rows to rotate, so the output keeps the input shape.
    q = torch.zeros(0, 2, 3, 4, dtype=torch.float64)
    q_rot, k_rot = _run(q, q, 1)
    assert q_rot.shape == q.shape, (
        "empty batch rule broken: a batch of zero rows must return a tensor of the same shape"
    )
    assert k_rot.shape == q.shape, (
        "empty batch rule broken: a batch of zero rows must return keys of the same shape"
    )


def test_gradients_flow_back_through_the_result():
    gen = torch.Generator().manual_seed(75)
    q = _real_values(gen, [1, 2, 3, 4]).requires_grad_()
    k = _real_values(gen, [1, 2, 3, 4]).requires_grad_()
    mix = _real_values(gen, [2, 1, 2, 3, 4])
    inputs = (q, k)
    try:
        out = torch.stack(apply_rope_qk(q, k, 3))
    except Exception:
        pytest.fail(
            "a valid call failed on tensors that require gradients: stay in PyTorch, with no "
            "NumPy or Python-number round trip",
            pytrace=False,
        )
    assert out.requires_grad, (
        "the result is cut off from the autograd graph: keep every step a torch operation, "
        "with no detach, no_grad, item or NumPy on the way"
    )
    got = torch.autograd.grad((out * mix).sum(), inputs, allow_unused=True)
    want = torch.autograd.grad(
        (torch.stack((_ref_half_split(q, 3), _ref_half_split(k, 3))) * mix).sum(),
        inputs,
        allow_unused=True,
    )
    for g, w in zip(got, want, strict=True):
        expected = torch.zeros((), dtype=torch.float64) if w is None else w
        have = torch.zeros_like(expected) if g is None else g
        assert torch.allclose(have, expected, rtol=1e-7, atol=1e-9), (
            "a gradient through the result is wrong or missing: an input or a statistic was "
            "taken out of the autograd graph"
        )
