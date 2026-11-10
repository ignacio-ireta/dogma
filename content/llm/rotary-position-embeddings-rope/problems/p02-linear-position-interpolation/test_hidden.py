# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show hidden
# inputs, shapes or expected values. The oracle uses complex arithmetic with torch.polar.
from __future__ import annotations

import copy

import pytest
import torch
from submission import apply_rope_scaled

ATOL = 1e-10
RTOL = 1e-10
QUERY_HINT = "check the scaled position (offset + t) / scale and theta_i = 10000 ** (-2 i / d)"


def _phi(tokens, d, offset, scale):
    """Angle table phi[t, i] = ((offset + t) / scale) * 10000**(-2 i / d), float64."""
    t = (torch.arange(offset, offset + tokens, dtype=torch.float64) / scale).unsqueeze(1)
    i = torch.arange(d // 2, dtype=torch.float64).unsqueeze(0)
    return t * torch.pow(torch.tensor(10000.0, dtype=torch.float64), -2.0 * i / d)


def _ref(x, offset, scale):
    """Complex reference: z_i = x_i + i x_{i+d/2}, multiplied by exp(i phi), split back."""
    half = x.shape[-1] // 2
    x64 = x.double()
    z = torch.complex(x64[..., :half], x64[..., half:])
    phi = _phi(x.shape[-2], x.shape[-1], offset, scale)
    rot = z * torch.polar(torch.ones_like(phi), phi)
    return torch.cat([rot.real, rot.imag], dim=-1).to(x.dtype)


def _run(q, k, offset, scale, *, pass_through=False):
    """Call the learner on copies, so an in-place write cannot corrupt the oracle's inputs."""
    call_q, call_k = (q, k) if pass_through else (q.clone(), k.clone())
    try:
        out = apply_rope_scaled(call_q, call_k, offset, scale)
    except ValueError:
        pytest.fail("valid input was rejected: check the contract")
    except Exception as exc:
        pytest.fail(
            f"valid input crashed ({type(exc).__name__}): check the (B, H, T, d) indexing and "
            "that the scaled positions broadcast over the batch and head axes"
        )
    if not isinstance(out, tuple) or len(out) != 2:
        pytest.fail("return exactly two tensors, (q_rot, k_rot): the value tensor is never rotated")
    for tensor, source in zip(out, (q, k), strict=True):
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != tuple(source.shape):
            pytest.fail("each rotated tensor must keep the input shape (B, H, T, d)")
        if tensor.dtype != source.dtype:
            pytest.fail("each rotated tensor must keep the input dtype")
        if not torch.isfinite(tensor).all():
            pytest.fail("non-finite rotation: check the scaled positions for large offsets")
    return out


@pytest.mark.parametrize(
    "shape, offset, scale",
    [((1, 1, 1, 2), 0, 1), ((2, 3, 4, 8), 0, 2), ((2, 2, 5, 6), 3, 2.5), ((1, 4, 7, 16), 11, 4),
     ((3, 1, 2, 4), 100, 0.5)],
    ids=["single-token-two-wide", "batched-scale-two", "offset-fractional-scale",
         "eight-wide-scale-four", "stretched-positions"],
)  # fmt: skip
def test_matches_complex_reference_for_several_scales(shape, offset, scale):
    gen = torch.Generator().manual_seed(sum(shape) + offset)
    q = torch.randn(*shape, generator=gen, dtype=torch.float64)
    k = torch.randn(*shape, generator=gen, dtype=torch.float64)
    q_rot, k_rot = _run(q, k, offset, scale)
    if not torch.allclose(q_rot, _ref(q, offset, scale), atol=ATOL, rtol=RTOL):
        pytest.fail(f"rotated queries differ from the reference: {QUERY_HINT}")
    if not torch.allclose(k_rot, _ref(k, offset, scale), atol=ATOL, rtol=RTOL):
        pytest.fail("rotated keys differ from the reference: keys need the same scale as queries")


def test_scale_one_matches_the_unscaled_rotation():
    gen = torch.Generator().manual_seed(21)
    q = torch.randn(2, 2, 4, 6, generator=gen, dtype=torch.float64)
    k = torch.randn(2, 2, 4, 6, generator=gen, dtype=torch.float64)
    q_rot, _ = _run(q, k, 5, 1)
    if not torch.allclose(q_rot, _ref(q, 5, 1), atol=ATOL, rtol=RTOL):
        pytest.fail("scale 1 must reproduce the unscaled rotation: check the division by scale")


def test_lowest_pair_rotates_by_offset_plus_t_over_scale():
    # With d = 2 there is one pair and theta_0 = 1, so the angle is exactly (offset + t) / scale.
    # A changed base, which keeps theta_0 = 1 but alters others, cannot pass this test by accident.
    x = torch.tensor(
        [[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float64
    )
    x = x.reshape(1, 1, 4, 2)
    q_rot, _ = _run(x, x, 1, 4)
    angles = torch.tensor([(1 + t) / 4 for t in range(4)], dtype=torch.float64)
    expected = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    if not torch.allclose(q_rot[0, 0], expected, atol=ATOL, rtol=RTOL):
        pytest.fail("pair 0 must rotate by (offset + t) / scale: the positions are not scaled")


def test_base_is_not_changed_by_interpolation():
    # A method that scales the base instead of the positions changes the higher pairs.
    x = torch.ones(1, 1, 3, 6, dtype=torch.float64)
    q_rot, _ = _run(x, x, 0, 2)
    if not torch.allclose(q_rot, _ref(x, 0, 2), atol=ATOL, rtol=RTOL):
        pytest.fail(
            "higher pairs disagree with the reference: keep the base 10000 and scale positions"
        )


def test_query_key_scores_depend_only_on_the_scaled_difference():
    gen = torch.Generator().manual_seed(23)
    q = torch.randn(1, 2, 5, 8, generator=gen, dtype=torch.float64)
    k = torch.randn(1, 2, 5, 8, generator=gen, dtype=torch.float64)
    q0, k0 = _run(q, k, 0, 3)
    q9, k9 = _run(q, k, 9, 3)
    if not torch.allclose(
        q0 @ k0.transpose(-1, -2), q9 @ k9.transpose(-1, -2), atol=1e-9, rtol=1e-9
    ):
        pytest.fail("scores change under a common shift: the offset must move q and k alike")


def test_large_offset_and_stretched_positions_match_the_reference():
    # Angles near 1e6 radians carry float64 round-off of about 1e-10 from the frequency table,
    # so this one check uses atol 1e-8. That is still far below any real-model error.
    gen = torch.Generator().manual_seed(25)
    q = torch.randn(1, 2, 3, 8, generator=gen, dtype=torch.float64)
    k = torch.randn(1, 2, 3, 8, generator=gen, dtype=torch.float64)
    q_rot, k_rot = _run(q, k, 10**6, 0.5)
    if not torch.allclose(q_rot, _ref(q, 10**6, 0.5), atol=1e-8, rtol=1e-8):
        pytest.fail(f"rotated queries differ at a large offset: {QUERY_HINT}")
    if not torch.allclose(k_rot, _ref(k, 10**6, 0.5), atol=1e-8, rtol=1e-8):
        pytest.fail("rotated keys differ at a large offset: keys need the same scale as queries")


@pytest.mark.parametrize("magnitude", [1e6, 1e-6], ids=["large-magnitude", "small-magnitude"])
def test_extreme_input_magnitudes_match_the_reference(magnitude):
    gen = torch.Generator().manual_seed(27)
    q = magnitude * torch.randn(1, 1, 4, 6, generator=gen, dtype=torch.float64)
    q_rot, _ = _run(q, q, 2, 2.0)
    if not torch.allclose(q_rot, _ref(q, 2, 2.0), atol=ATOL, rtol=RTOL):
        pytest.fail(f"rotated queries differ at this magnitude: {QUERY_HINT}")


def test_float32_inputs_return_float32_close_to_reference():
    gen = torch.Generator().manual_seed(29)
    q64 = torch.randn(2, 2, 4, 8, generator=gen, dtype=torch.float64)
    q_rot, k_rot = _run(q64.float(), q64.float(), 6, 2.0)
    if q_rot.dtype != torch.float32 or k_rot.dtype != torch.float32:
        pytest.fail("float32 inputs must return float32 tensors")
    if not torch.allclose(q_rot.double(), _ref(q64, 6, 2.0), atol=1e-5, rtol=1e-5):
        pytest.fail("float32 result drifts from the reference beyond single-precision round-off")


def test_inputs_and_scale_are_not_mutated():
    gen = torch.Generator().manual_seed(31)
    q = torch.randn(1, 2, 3, 4, generator=gen, dtype=torch.float64)
    k = torch.randn(1, 2, 3, 4, generator=gen, dtype=torch.float64)
    q_before, k_before = copy.deepcopy(q), copy.deepcopy(k)
    _run(q, k, 2, 2.0, pass_through=True)
    if not torch.equal(q, q_before) or not torch.equal(k, k_before):
        pytest.fail("the input tensors were modified in place: write into new tensors")


def test_empty_sequence_returns_empty_tensors():
    q = torch.zeros(2, 2, 0, 4, dtype=torch.float64)
    q_rot, k_rot = _run(q, q, 3, 2.0)
    if q_rot.shape != q.shape or k_rot.shape != q.shape:
        pytest.fail("an empty sequence must return empty tensors of the same width")


def _ones(width, dims=4):
    shape = (1, 2, 3, width) if dims == 4 else (2, 3, width)
    return torch.ones(*shape, dtype=torch.float64)


def _mutant(**changes):
    """Return a function that builds one invalid call from the valid base call."""

    def build():
        q = torch.ones(1, 2, 3, 4, dtype=torch.float64)
        args = {"q": q, "k": q.clone(), "offset": 2, "scale": 2.0}
        args.update(changes)
        return args

    return build


INVALID_CASES = {
    "zero-scale": _mutant(scale=0.0),
    "negative-scale": _mutant(scale=-1.0),
    "nan-scale": _mutant(scale=float("nan")),
    "inf-scale": _mutant(scale=float("inf")),
    "bool-scale": _mutant(scale=True),
    "string-scale": _mutant(scale="2"),
    "odd-width": _mutant(q=_ones(5), k=_ones(5)),
    "negative-offset": _mutant(offset=-1),
    "float-offset": _mutant(offset=2.0),
    "dtype-mismatch": _mutant(k=torch.ones(1, 2, 3, 4, dtype=torch.float32)),
    "three-dimensional": _mutant(q=_ones(4, dims=3), k=_ones(4, dims=3)),
}  # fmt: skip


@pytest.mark.parametrize("case", list(INVALID_CASES), ids=list(INVALID_CASES))
def test_invalid_inputs_raise_value_error(case):
    args = INVALID_CASES[case]()
    try:
        apply_rope_scaled(args.pop("q"), args.pop("k"), args.pop("offset"), args.pop("scale"))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input ({case}) raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"invalid input ({case}) was accepted instead of raising ValueError")


def test_non_tensor_query_is_rejected_with_value_error():
    q = torch.ones(1, 2, 3, 4, dtype=torch.float64)
    try:
        apply_rope_scaled(q.tolist(), q, 2, 2.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a non-tensor query raised {type(exc).__name__} instead of ValueError")
    pytest.fail("a non-tensor query was accepted instead of raising ValueError")


def test_zero_feature_width_is_rejected_with_value_error():
    z = torch.zeros(1, 2, 3, 0, dtype=torch.float64)
    try:
        apply_rope_scaled(z, z, 2, 2.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a zero feature width raised {type(exc).__name__} instead of ValueError")
    pytest.fail("a zero feature width was accepted instead of raising ValueError")
