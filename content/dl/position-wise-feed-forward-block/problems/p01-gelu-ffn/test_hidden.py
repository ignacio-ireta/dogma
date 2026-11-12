# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import gelu_ffn as _learner

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
    key = f"p01-gelu-ffn:values:{draw}"
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


def _params(gen, d_model, d_ff):
    return (
        _real_values(gen, [d_ff, d_model], torch.float64),
        _real_values(gen, [d_ff], torch.float64),
        _real_values(gen, [d_model, d_ff], torch.float64),
        _real_values(gen, [d_model], torch.float64),
    )


def _oracle(x, W1, b1, W2, b2):
    """PyTorch reference: linear -> exact GELU (erf form) -> linear, applied per position."""
    h = torch.nn.functional.linear(x, W1, b1)
    h = torch.nn.functional.gelu(h, approximate="none")
    return torch.nn.functional.linear(h, W2, b2)


def test_matches_pytorch_reference_on_seeded_batches():
    gen = torch.Generator().manual_seed(41)
    for shape, d_model, d_ff in [((5, 4), 4, 16), ((2, 3, 6), 6, 24), ((7,), 3, 5)]:
        x = _real_values(gen, [*shape, d_model], torch.float64)
        W1, b1, W2, b2 = _params(gen, d_model, d_ff)
        torch.testing.assert_close(
            _call(x, W1, b1, W2, b2),
            _oracle(x, W1, b1, W2, b2),
            atol=1e-9,
            rtol=1e-9,
            msg="output disagrees with the reference FFN: check the layer order and the GELU",
        )


def test_gelu_uses_exact_erf_form_not_tanh_approximation():
    # Pick one hidden unit that is positive, and check against the exact value at that point.
    x = torch.tensor([[2.0]], dtype=torch.float64)
    one = torch.tensor([[1.0]], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    got = _call(x, one, zero, one, zero)
    exact = torch.nn.functional.gelu(torch.tensor([2.0], dtype=torch.float64), approximate="none")
    tanh_form = torch.nn.functional.gelu(
        torch.tensor([2.0], dtype=torch.float64), approximate="tanh"
    )
    assert abs(float(exact) - float(tanh_form)) > 1e-6, "test point must separate the two forms"
    torch.testing.assert_close(
        got,
        exact.reshape(1, 1),
        atol=1e-9,
        rtol=1e-9,
        msg="GELU must be the exact erf form, not the tanh approximation",
    )


def test_positions_are_independent():
    gen = torch.Generator().manual_seed(42)
    W1, b1, W2, b2 = _params(gen, 4, 9)
    x = _real_values(gen, [6, 4], torch.float64)
    base = _call(x, W1, b1, W2, b2)
    changed = x.clone()
    changed[2] += 5.0
    out = _call(changed, W1, b1, W2, b2)
    mask = torch.ones(6, dtype=torch.bool)
    mask[2] = False
    torch.testing.assert_close(
        out[mask],
        base[mask],
        atol=1e-12,
        rtol=0,
        msg="positions must not interact: changing one token changed another",
    )


def test_empty_hidden_layer_returns_output_bias():
    gen = torch.Generator().manual_seed(43)
    x = _real_values(gen, [3, 4], torch.float64)
    W1 = torch.zeros(0, 4, dtype=torch.float64)
    b1 = torch.zeros(0, dtype=torch.float64)
    W2 = torch.zeros(4, 0, dtype=torch.float64)
    b2 = _real_values(gen, [4], torch.float64)
    got = _call(x, W1, b1, W2, b2)
    torch.testing.assert_close(
        got,
        b2.expand(3, 4),
        atol=1e-12,
        rtol=0,
        msg="with an empty hidden layer the output must be b2 at every position",
    )


def test_non_square_weights_use_nn_linear_orientation():
    gen = torch.Generator().manual_seed(44)
    d_model, d_ff = 2, 5
    x = _real_values(gen, [4, d_model], torch.float64)
    W1, b1, W2, b2 = _params(gen, d_model, d_ff)
    torch.testing.assert_close(
        _call(x, W1, b1, W2, b2),
        _oracle(x, W1, b1, W2, b2),
        atol=1e-9,
        rtol=1e-9,
        msg="weights are in the wrong orientation: expansion x @ W1.T, projection h @ W2.T",
    )


def test_square_hidden_layer_uses_nn_linear_orientation():
    # With d_ff == d_model, x @ W1 runs without a shape error and gives a wrong answer
    # unless W1 is symmetric, so this case needs non-symmetric weights.
    gen = torch.Generator().manual_seed(45)
    d_model = d_ff = 3
    x = _real_values(gen, [4, d_model], torch.float64)
    W1, b1, W2, b2 = _params(gen, d_model, d_ff)
    torch.testing.assert_close(
        _call(x, W1, b1, W2, b2),
        _oracle(x, W1, b1, W2, b2),
        atol=1e-9,
        rtol=1e-9,
        msg="square weights are in the wrong orientation: x @ W1 instead of x @ W1.T",
    )


def test_expansion_bias_is_added_before_the_nonlinearity():
    # With W1 = 0, the hidden pre-activation is b1 for every position, so GELU(b1) is the hidden
    # value and the output is W2 @ GELU(b1) + b2.
    gen = torch.Generator().manual_seed(45)
    d_model, d_ff = 3, 4
    x = _real_values(gen, [2, d_model], torch.float64)
    W1 = torch.zeros(d_ff, d_model, dtype=torch.float64)
    b1 = _real_values(gen, [d_ff], torch.float64)
    _, _, W2, b2 = _params(gen, d_model, d_ff)
    hidden = torch.nn.functional.gelu(b1, approximate="none")
    want = hidden @ W2.T + b2
    torch.testing.assert_close(
        _call(x, W1, b1, W2, b2),
        want.expand(2, d_model),
        atol=1e-9,
        rtol=1e-9,
        msg="the first bias must be added before the GELU, not after it",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False))
def test_single_unit_matches_reference_across_range(value):
    x = torch.tensor([[value]], dtype=torch.float64)
    one = torch.tensor([[1.0]], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    torch.testing.assert_close(
        _call(x, one, zero, one, zero),
        _oracle(x, one, zero, one, zero),
        atol=1e-9,
        rtol=1e-9,
        msg="exact GELU must match the reference on the full range, negative tail included",
    )


def test_large_magnitudes_stay_finite_and_match_reference():
    gen = torch.Generator().manual_seed(46)
    d_model, d_ff = 3, 6
    x = _real_values(gen, [4, d_model], torch.float64) * 1e3
    W1, b1, W2, b2 = _params(gen, d_model, d_ff)
    out = _call(x, W1, b1, W2, b2)
    assert torch.isfinite(out).all(), "large inputs must not produce NaN or infinity"
    torch.testing.assert_close(
        out,
        _oracle(x, W1, b1, W2, b2),
        atol=1e-6,
        rtol=1e-9,
        msg="large inputs disagree with the reference: check the tails of the GELU",
    )


def test_small_magnitudes_match_reference():
    gen = torch.Generator().manual_seed(47)
    d_model, d_ff = 3, 6
    x = _real_values(gen, [4, d_model], torch.float64) * 1e-9
    W1, b1, W2, b2 = _params(gen, d_model, d_ff)
    W1 = W1 * 1e-9
    b1 = b1 * 1e-9
    torch.testing.assert_close(
        _call(x, W1, b1, W2, b2),
        _oracle(x, W1, b1, W2, b2),
        atol=1e-18,
        rtol=1e-9,
        msg="tiny inputs must keep the local linear behaviour of the exact GELU",
    )


def test_output_dtype_and_shape_follow_input():
    gen = torch.Generator().manual_seed(48)
    x = _real_values(gen, [2, 5, 3], torch.float64)
    W1, b1, W2, b2 = _params(gen, 3, 7)
    out = _call(x, W1, b1, W2, b2)
    assert out.dtype == torch.float64, "output dtype must be float64"
    assert tuple(out.shape) == (2, 5, 3), "output shape must equal the input shape (..., d_model)"


def test_scalar_input_raises_value_error():
    gen = torch.Generator().manual_seed(49)
    W1, b1, W2, b2 = _params(gen, 2, 3)
    _invalid(torch.tensor(1.0, dtype=torch.float64), W1, b1, W2, b2)


def test_feature_mismatch_raises_value_error():
    gen = torch.Generator().manual_seed(50)
    W1, b1, W2, b2 = _params(gen, 2, 3)
    _invalid(torch.zeros(4, 3, dtype=torch.float64), W1, b1, W2, b2)


def test_wrong_hidden_bias_length_raises_value_error():
    gen = torch.Generator().manual_seed(51)
    x = torch.zeros(4, 2, dtype=torch.float64)
    W1, _, W2, b2 = _params(gen, 2, 3)
    _invalid(x, W1, torch.zeros(2, dtype=torch.float64), W2, b2)


def test_wrong_output_bias_length_raises_value_error():
    gen = torch.Generator().manual_seed(52)
    x = torch.zeros(4, 2, dtype=torch.float64)
    W1, b1, W2, _ = _params(gen, 2, 3)
    _invalid(x, W1, b1, W2, torch.zeros(3, dtype=torch.float64))


def test_inputs_are_not_mutated():
    gen = torch.Generator().manual_seed(53)
    x = _real_values(gen, [3, 2], torch.float64)
    W1, b1, W2, b2 = _params(gen, 2, 4)
    before = [t.clone() for t in (x, W1, b1, W2, b2)]
    _call(x, W1, b1, W2, b2)
    for name, now, then in zip(
        ["X", "W1", "b1", "W2", "b2"], (x, W1, b1, W2, b2), before, strict=True
    ):
        assert torch.equal(now, then), f"{name} must not be modified in place"


def test_single_position_without_leading_axes_is_accepted():
    gen = torch.Generator().manual_seed(54)
    W1, b1, W2, b2 = _params(gen, 3, 4)
    x = _real_values(gen, [3], torch.float64)  # one position, shape (d_model,)
    out = _call(x, W1, b1, W2, b2)
    assert tuple(out.shape) == (3,), "a 1-D input is one position and must return shape (d_model,)"
    torch.testing.assert_close(
        out,
        _oracle(x.unsqueeze(0), W1, b1, W2, b2).squeeze(0),
        atol=1e-9,
        rtol=1e-9,
        msg="a single position with no leading axes must give the same values as a batch of one",
    )
