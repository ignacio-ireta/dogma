# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import swiglu_ffn as _learner

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
    key = f"p02-swiglu-ffn:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def _call(*args):
    """Call the learner; any exception on a valid input is reported as a named contract failure."""
    try:
        return _learner(*args)
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _invalid(*args):
    try:
        _learner(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid argument was accepted instead of raising ValueError")


def _weights(gen, d_model, d_ff):
    return (
        _real_values(gen, [d_ff, d_model], torch.float64),
        _real_values(gen, [d_ff, d_model], torch.float64),
        _real_values(gen, [d_model, d_ff], torch.float64),
    )


def _oracle(x, W, V, W2):
    """PyTorch reference: silu(x W^T) * (x V^T), then the down projection (no biases)."""
    gate = torch.nn.functional.linear(x, W)
    up = torch.nn.functional.linear(x, V)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, W2)


def test_matches_pytorch_reference_on_seeded_batches():
    gen = torch.Generator().manual_seed(61)
    for shape, d_model, d_ff in [((4,), 3, 8), ((2, 5), 4, 6), ((3, 2, 4), 4, 10)]:
        x = _real_values(gen, [*shape, d_model], torch.float64)
        W, V, W2 = _weights(gen, d_model, d_ff)
        torch.testing.assert_close(
            _call(x, W, V, W2),
            _oracle(x, W, V, W2),
            atol=1e-9,
            rtol=1e-9,
            msg="output disagrees with the reference SwiGLU: check the gate, up and down paths",
        )


def test_silu_applies_to_the_gate_path_not_the_up_path():
    gen = torch.Generator().manual_seed(62)
    d_model, d_ff = 3, 5
    x = _real_values(gen, [4, d_model], torch.float64)
    W, V, W2 = _weights(gen, d_model, d_ff)
    torch.testing.assert_close(
        _call(x, W, V, W2),
        _oracle(x, W, V, W2),
        atol=1e-9,
        rtol=1e-9,
        msg="the activation must sit on the gate path: swapping W and V changed the output",
    )
    swapped = _call(x, V, W, W2)
    assert not torch.allclose(swapped, _call(x, W, V, W2), atol=1e-6), (
        "gate and up roles look interchangeable: the activation is not on the gate"
    )


def test_activation_is_silu_not_relu_or_gelu():
    # With gate = up = 1.5, the output is SiLU(1.5) * 1.5, which differs from ReLU(1.5) * 1.5 and
    # from GELU(1.5) * 1.5 at this point.
    x = torch.tensor([[1.5]], dtype=torch.float64)
    one = torch.tensor([[1.0]], dtype=torch.float64)
    got = _call(x, one, one, one)
    want = _oracle(x, one, one, one)
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="the gate must use SiLU = z * sigmoid(z): ReLU or GELU give a different value",
    )


def test_weights_use_nn_linear_orientation_on_non_square_matrices():
    gen = torch.Generator().manual_seed(64)
    d_model, d_ff = 2, 7
    x = _real_values(gen, [5, d_model], torch.float64)
    W, V, W2 = _weights(gen, d_model, d_ff)
    torch.testing.assert_close(
        _call(x, W, V, W2),
        _oracle(x, W, V, W2),
        atol=1e-9,
        rtol=1e-9,
        msg="weights are in the wrong orientation: gate and up are x @ W.T and x @ V.T",
    )


def test_positions_are_independent():
    gen = torch.Generator().manual_seed(65)
    W, V, W2 = _weights(gen, 3, 6)
    x = _real_values(gen, [5, 3], torch.float64)
    base = _call(x, W, V, W2)
    changed = x.clone()
    changed[1] *= -4.0
    out = _call(changed, W, V, W2)
    keep = torch.tensor([True, False, True, True, True])
    torch.testing.assert_close(
        out[keep],
        base[keep],
        atol=1e-12,
        rtol=0,
        msg="positions must not interact: changing one token changed another",
    )


def test_empty_gate_returns_zeros():
    gen = torch.Generator().manual_seed(66)
    x = _real_values(gen, [2, 3], torch.float64)
    empty = torch.zeros(0, 3, dtype=torch.float64)
    W2 = torch.zeros(3, 0, dtype=torch.float64)
    out = _call(x, empty, empty, W2)
    assert tuple(out.shape) == (2, 3), "an empty gate must still return shape (..., d_model)"
    assert float(out.abs().max()) == 0.0, "an empty gate must give zeros"


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False))
def test_single_gate_value_matches_reference_across_range(value):
    x = torch.tensor([[value]], dtype=torch.float64)
    one = torch.tensor([[1.0]], dtype=torch.float64)
    torch.testing.assert_close(
        _call(x, one, one, one),
        _oracle(x, one, one, one),
        atol=1e-9,
        rtol=1e-9,
        msg="SiLU must match the reference on the whole range, including large negative gates",
    )


def test_large_magnitudes_stay_finite():
    gen = torch.Generator().manual_seed(67)
    W, V, W2 = _weights(gen, 3, 6)
    x = _real_values(gen, [4, 3], torch.float64) * 1e3
    out = _call(x, W, V, W2)
    assert torch.isfinite(out).all(), "large inputs must not produce NaN or infinity"
    torch.testing.assert_close(
        out,
        _oracle(x, W, V, W2),
        atol=1e-6,
        rtol=1e-9,
        msg="large inputs disagree with the reference: check the sigmoid on the gate",
    )


def test_small_magnitudes_match_reference():
    gen = torch.Generator().manual_seed(68)
    W, V, W2 = _weights(gen, 3, 6)
    x = _real_values(gen, [4, 3], torch.float64) * 1e-9
    torch.testing.assert_close(
        _call(x, W, V, W2),
        _oracle(x, W, V, W2),
        atol=1e-18,
        rtol=1e-9,
        msg="tiny inputs must follow the same formula: check the gate and the product",
    )


def test_output_dtype_and_shape_follow_input():
    gen = torch.Generator().manual_seed(69)
    W, V, W2 = _weights(gen, 4, 9)
    x = _real_values(gen, [2, 5, 4], torch.float64)
    out = _call(x, W, V, W2)
    assert out.dtype == torch.float64, "output dtype must be float64"
    assert tuple(out.shape) == (2, 5, 4), "output shape must equal (..., d_model)"


def test_scalar_input_raises_value_error():
    gen = torch.Generator().manual_seed(70)
    W, V, W2 = _weights(gen, 2, 3)
    _invalid(torch.tensor(1.0, dtype=torch.float64), W, V, W2)


def test_feature_mismatch_raises_value_error():
    gen = torch.Generator().manual_seed(71)
    W, V, W2 = _weights(gen, 2, 3)
    _invalid(torch.zeros(4, 3, dtype=torch.float64), W, V, W2)


def test_wrong_down_projection_shape_raises_value_error():
    gen = torch.Generator().manual_seed(72)
    W, V, _ = _weights(gen, 2, 3)
    _invalid(torch.zeros(4, 2, dtype=torch.float64), W, V, torch.zeros(3, 2, dtype=torch.float64))


def test_inputs_are_not_mutated():
    gen = torch.Generator().manual_seed(73)
    x = _real_values(gen, [3, 2], torch.float64)
    W, V, W2 = _weights(gen, 2, 4)
    before = [t.clone() for t in (x, W, V, W2)]
    _call(x, W, V, W2)
    for name, now, then in zip(["X", "W", "V", "W2"], (x, W, V, W2), before, strict=True):
        assert torch.equal(now, then), f"{name} must not be modified in place"


def test_one_dimensional_input_is_a_single_position():
    gen = torch.Generator().manual_seed(74)
    d_model, d_ff = 3, 5
    x = _real_values(gen, [d_model], torch.float64)
    W, V, W2 = _weights(gen, d_model, d_ff)
    out = _call(x, W, V, W2)
    assert tuple(out.shape) == (d_model,), (
        "a single position with shape (d_model,) must return shape (d_model,)"
    )
    torch.testing.assert_close(
        out,
        _oracle(x, W, V, W2),
        atol=1e-9,
        rtol=1e-9,
        msg="a 1-D input is valid: only a 0-dimensional X must be rejected",
    )
