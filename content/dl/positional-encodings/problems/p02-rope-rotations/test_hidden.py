# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import apply_rope as _learner


def _rope(x):
    """Call the learner; any exception on a valid input is reported as a named contract failure."""
    try:
        return _learner(x)
    except Exception as exc:
        pytest.fail(f"valid input violated the contract: raised {type(exc).__name__}")


def _oracle(x):
    """Independent reference: rotate each adjacent pair as a complex number by e^(i t theta)."""
    seq_len, d = x.shape[-2], x.shape[-1]
    half = d // 2
    t = torch.arange(seq_len, dtype=torch.float64).unsqueeze(1)
    theta = 10000.0 ** (-2.0 * torch.arange(half, dtype=torch.float64) / d)
    rot = torch.polar(torch.ones(seq_len, half, dtype=torch.float64), t * theta)
    z = torch.view_as_complex(x.double().reshape(*x.shape[:-1], half, 2).contiguous())
    return torch.view_as_real(z * rot).reshape(x.shape)


def _invalid(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid input was accepted instead of raising ValueError")


def test_matches_complex_multiplication_reference_on_seeded_batches():
    gen = torch.Generator().manual_seed(7)
    for shape in [(2, 5, 8), (3, 7, 4), (6, 2)]:
        x = torch.randn(*shape, generator=gen, dtype=torch.float64)
        torch.testing.assert_close(
            _rope(x),
            _oracle(x),
            atol=1e-9,
            rtol=1e-9,
            msg="rotation disagrees with the complex-pair reference: check the pair layout",
        )


def test_adjacent_pairs_are_rotated_not_split_halves():
    # With d = 4 the adjacent pairs are (0, 1) and (2, 3); a split-half layout pairs (0, 2), (1, 3).
    x = torch.zeros(3, 4, dtype=torch.float64)
    x[2, 0] = 1.0  # only feature 0 is set in row 2
    out = _rope(x)
    assert torch.allclose(out[2, 2:], torch.zeros(2, dtype=torch.float64)), (
        "a single feature in pair 0 must stay inside pair 0: the layout pairs features in halves"
    )


def test_sine_sign_and_angle_at_position_one_for_first_pair():
    # theta_0 = 1, so at position 1 the pair (1, 0) becomes (cos 1, sin 1).
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    out = _rope(x)
    one = torch.tensor(1.0, dtype=torch.float64)
    expected = torch.stack([torch.cos(one), torch.sin(one)])
    torch.testing.assert_close(
        out[1],
        expected,
        atol=1e-9,
        rtol=1e-9,
        msg="sign or angle error at position 1: the sine term must carry the opposite sign",
    )


def test_position_zero_is_identity_for_any_values():
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(4, 6, 8, generator=gen, dtype=torch.float64)
    out = _rope(x)
    torch.testing.assert_close(
        out[..., 0, :],
        x[..., 0, :],
        atol=1e-12,
        rtol=0,
        msg="position 0 must be the identity: a rotation by zero angle changes nothing",
    )


def test_each_row_keeps_its_euclidean_norm():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(2, 9, 6, generator=gen, dtype=torch.float64)
    out = _rope(x)
    torch.testing.assert_close(
        out.norm(dim=-1),
        x.norm(dim=-1),
        atol=1e-9,
        rtol=1e-9,
        msg="rotation must preserve each row norm: a scaling or non-orthogonal step is wrong",
    )


def test_query_key_dot_product_depends_only_on_relative_offset():
    d, length = 6, 10
    gen = torch.Generator().manual_seed(2)
    q_vec = torch.randn(d, generator=gen, dtype=torch.float64)
    k_vec = torch.randn(d, generator=gen, dtype=torch.float64)
    q = _rope(q_vec.expand(length, d).contiguous())
    k = _rope(k_vec.expand(length, d).contiguous())
    scores = q @ k.T  # scores[m, n] = (rotated q at m) . (rotated k at n)
    for m, n, shift in [(0, 3, 2), (1, 4, 4), (2, 2, 5), (3, 1, 3)]:
        torch.testing.assert_close(
            scores[m + shift, n + shift],
            scores[m, n],
            atol=1e-9,
            rtol=1e-9,
            msg="query-key score must depend only on m - n: check both q and k rotate",
        )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=1, max_value=6), st.integers(min_value=1, max_value=5))
def test_leading_batch_axes_are_independent(batch, half):
    gen = torch.Generator().manual_seed(batch * 10 + half)
    x = torch.randn(batch, 4, 2 * half, generator=gen, dtype=torch.float64)
    together = _rope(x)
    separately = torch.stack([_rope(x[b]) for b in range(batch)])
    torch.testing.assert_close(
        together,
        separately,
        atol=1e-12,
        rtol=0,
        msg="batch rows must be rotated independently: check the position axis",
    )


def test_empty_sequence_returns_empty_tensor_with_same_width():
    x = torch.zeros(2, 0, 4, dtype=torch.float64)
    out = _rope(x)
    assert tuple(out.shape) == (2, 0, 4), "an empty sequence must return shape (..., 0, d)"


def test_large_positions_match_reference():
    x = torch.ones(4096, 16, dtype=torch.float64)
    torch.testing.assert_close(
        _rope(x),
        _oracle(x),
        atol=1e-9,
        rtol=1e-9,
        msg="long sequences: check the position-times-frequency angle table",
    )


def test_small_magnitudes_scale_linearly():
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    scale = 1e-150
    torch.testing.assert_close(
        _rope(x * scale),
        _rope(x) * scale,
        atol=0,
        rtol=1e-9,
        msg="rotation must be linear: a normalisation or clamp changes tiny inputs",
    )


def test_output_dtype_and_shape_follow_input():
    x64 = torch.ones(3, 2, 4, dtype=torch.float64)
    assert _rope(x64).dtype == torch.float64, "float64 input must return float64"
    x32 = torch.ones(3, 2, 4, dtype=torch.float32)
    out32 = _rope(x32)
    assert out32.dtype == torch.float32, "float32 input must return float32 (same dtype as x)"
    assert tuple(out32.shape) == (3, 2, 4), "output shape must equal input shape"


def test_one_dimensional_input_raises_value_error():
    _invalid(_learner, torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("width", [3, 5], ids=["odd-three", "odd-five"])
def test_odd_feature_width_raises_value_error(width):
    _invalid(_learner, torch.ones(2, width, dtype=torch.float64))


def test_zero_feature_width_raises_value_error():
    _invalid(_learner, torch.ones(2, 0, dtype=torch.float64))


def test_input_is_not_mutated():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(3, 5, 6, generator=gen, dtype=torch.float64)
    before = x.clone()
    _rope(x)
    assert torch.equal(x, before), "the input tensor must not be modified in place"
