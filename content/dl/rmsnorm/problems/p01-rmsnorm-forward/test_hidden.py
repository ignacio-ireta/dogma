# Hidden tests: oracle comparisons against PyTorch's functional RMSNorm, plus edge cases.
# The oracle is used only here; the submission may not use torch.nn or torch.rms_norm.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import rms_norm

from dogma_data import rng, sample_rows
from dogma_data import rng as data_rng

ATOL = 1e-9
RTOL = 1e-9


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
    key = f"p01-rmsnorm-forward:values:{draw}"
    rows = max(1, min(569, -(-total // 30)))
    drawn, _ = sample_rows("breast_cancer", data_rng(key), rows, standardize=True)
    flat = torch.tensor(drawn.ravel(), dtype=torch.float64)
    flat = flat.repeat(-(-total // flat.numel()))[:total]
    return flat.reshape(shape).to(dtype)


def _features(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    """Real inputs of the given shape: standardised breast_cancer features, scaled and shifted.

    The last axis holds the features and the leading axes hold the rows drawn. The columns
    are correlated and skewed, and a small batch can repeat a value, as measured data does.
    Weights, gains, biases and running statistics are parameters and stay random.
    """
    shape = tuple(shape)
    rows = math.prod(shape[:-1])
    drawn, _ = sample_rows(
        "breast_cancer", rng(f"rmsnorm-p01:features:{seed}"), rows, shape[-1], standardize=True
    )
    return torch.tensor(drawn.reshape(shape), dtype=torch.float64).to(dtype) * scale + offset


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return _real_values(g, [shape], dtype) * scale + offset


def _call(x, gain, eps=1e-6):
    snapshot = [x.clone(), gain.clone()]
    try:
        out = rms_norm(x, gain, eps=eps)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, dtypes and the contract",
            pytrace=False,
        )
    for name, before, after in zip(("x", "gain"), snapshot, (x, gain), strict=True):
        if not torch.equal(before, after):
            pytest.fail(f"input {name} was modified in place; return a new tensor", pytrace=False)
    if not torch.is_tensor(out):
        pytest.fail("return a torch tensor, not a NumPy array or a list", pytrace=False)
    if out.dtype != x.dtype:
        pytest.fail("output must keep the dtype of x; do not downcast", pytrace=False)
    return out


def _oracle(x, gain, eps=1e-6):
    return ref_f.rms_norm(x, (x.shape[-1],), gain, eps)


def _close(actual, expected, message, atol=ATOL, rtol=RTOL):
    if tuple(actual.shape) != tuple(expected.shape):
        pytest.fail("output shape must equal the input shape", pytrace=False)
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        pytest.fail(message, pytrace=False)


SHAPES = [(3,), (2, 4), (2, 3, 5), (1, 2, 3, 4), (4, 1), (6, 2)]


def test_matches_reference_over_random_leading_shapes():
    for seed, shape in enumerate(SHAPES):
        d = shape[-1]
        x = _features(shape, seed, scale=2.0, offset=0.5)
        gain = _randn((d,), seed + 10)
        eps = [1e-6, 1e-3, 1e-8][seed % 3]
        _close(
            _call(x, gain, eps=eps),
            _oracle(x, gain, eps=eps),
            "output differs from the reference RMSNorm: compute gain * x / sqrt(mean(x**2) + eps): "
            "no mean subtraction, eps inside the root, the mean taken over the last dimension, "
            "and the gain multiplying each feature",
        )


def test_mean_is_not_subtracted():
    # A constant offset must change the output. Centering would make both calls identical.
    x = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    gain = torch.ones(2, dtype=torch.float64)
    _close(
        _call(x, gain, eps=1e-12),
        torch.tensor([[0.4472135955, 1.3416407865]], dtype=torch.float64),
        "the mean was subtracted: RMSNorm divides by the root mean square and keeps the offset",
    )


def test_offset_example_is_not_invariant_to_a_constant_shift():
    x = _features((3, 4), 21, scale=1.0)
    gain = torch.ones(4, dtype=torch.float64)
    shifted = _call(x + 5.0, gain, eps=1e-12)
    original = _call(x, gain, eps=1e-12)
    if torch.allclose(shifted, original, atol=1e-6):
        pytest.fail(
            "the mean was subtracted: a constant shift of the input left the output unchanged",
            pytrace=False,
        )
    _close(
        shifted,
        _oracle(x + 5.0, gain, eps=1e-12),
        "shifted-input output differs from the reference RMSNorm",
    )


def test_eps_is_added_inside_the_square_root():
    # Mean square is 1e-6 and eps is 1e-6: placing eps outside changes the scale by about 700 times.
    x = torch.tensor([[1e-3, 1e-3]], dtype=torch.float64)
    gain = torch.ones(2, dtype=torch.float64)
    _close(
        _call(x, gain, eps=1e-6),
        _oracle(x, gain, eps=1e-6),
        "eps is added outside the square root; add it to the mean square before taking sqrt",
    )


def test_unit_mean_square_before_gain():
    x = _features((5, 6), 31, scale=4.0)
    gain = torch.ones(6, dtype=torch.float64)
    y = _call(x, gain, eps=1e-12)
    mean_square = (y * y).mean(dim=-1)
    _close(
        mean_square,
        torch.ones(5, dtype=torch.float64),
        "output mean square is not one: divide by the root mean square of each example",
        atol=1e-9,
    )


def test_statistic_runs_over_the_last_axis():
    x = _features((3, 5), 41, scale=2.0)
    gain = _randn((5,), 42)
    _close(
        _call(x, gain, eps=1e-12),
        _oracle(x, gain, eps=1e-12),
        "statistic taken over the wrong axis: the root mean square is per example, "
        "over the last dimension",
    )


def test_each_example_is_normalized_independently():
    x = _features((4, 3), 51, scale=3.0)
    gain = _randn((3,), 52)
    together = _call(x, gain)
    for i in range(x.shape[0]):
        _close(
            together[i : i + 1],
            _call(x[i : i + 1], gain),
            "an example's output depends on other examples in the batch",
        )


def test_large_magnitudes_match_reference():
    x = _features((3, 4), 61, scale=1e100)
    gain = _randn((4,), 62)
    y = _call(x, gain, eps=1e-6)
    _close(
        y, _oracle(x, gain, eps=1e-6), "large-magnitude output differs from the reference RMSNorm"
    )


def test_tiny_magnitudes_match_reference():
    x = _features((3, 4), 71, scale=1e-15)
    gain = _randn((4,), 72)
    y = _call(x, gain, eps=1e-30)
    _close(
        y, _oracle(x, gain, eps=1e-30), "tiny-magnitude output differs from the reference RMSNorm"
    )


def test_zero_rows_give_zero_output():
    x = torch.zeros(2, 3, dtype=torch.float64)
    gain = torch.ones(3, dtype=torch.float64)
    y = _call(x, gain, eps=1e-6)
    assert torch.isfinite(y).all(), (
        "zero input produced NaN or inf: eps must keep the division finite"
    )
    _close(y, torch.zeros_like(y), "zero input must give zero output")


def test_float32_inputs_return_float32():
    x = _features((3, 5), 81, dtype=torch.float32)
    gain = torch.ones(5, dtype=torch.float32)
    y = _call(x, gain, eps=1e-6)
    assert y.dtype == torch.float32, "output dtype must follow the input dtype"
    _close(
        y,
        _oracle(x, gain, eps=1e-6),
        "float32 output differs from the reference beyond float32 precision",
        atol=1e-5,
        rtol=1e-5,
    )


def test_empty_leading_dimension_returns_empty_output():
    x = torch.empty((0, 4), dtype=torch.float64)
    y = _call(x, torch.ones(4, dtype=torch.float64))
    assert tuple(y.shape) == (0, 4), "an empty batch must return an empty output of the same shape"


def test_invalid_arguments_raise_value_error():
    gain = torch.ones(3, dtype=torch.float64)
    x = _features((2, 3), 91)
    cases = {
        "a 0-dimensional x must raise ValueError": lambda: rms_norm(torch.tensor(1.0), gain),
        "an integer x must raise ValueError": lambda: rms_norm(
            torch.ones(2, 3, dtype=torch.int64), gain
        ),
        "a gain with the wrong length must raise ValueError": lambda: rms_norm(
            x, torch.ones(2, dtype=torch.float64)
        ),
        "a zero-width last dimension must raise ValueError": lambda: rms_norm(
            torch.empty(2, 0, dtype=torch.float64), torch.ones(0, dtype=torch.float64)
        ),
        "a zero eps must raise ValueError": lambda: rms_norm(x, gain, eps=0.0),
    }
    for message, thunk in cases.items():
        try:
            thunk()
        except ValueError:
            continue
        except Exception:
            pytest.fail(
                "invalid input must raise ValueError, not another exception type", pytrace=False
            )
        pytest.fail(message, pytrace=False)


def test_inputs_are_not_mutated():
    x = _features((3, 4), 101)
    gain = _randn((4,), 102)
    _call(x, gain)  # _call compares the arguments before and after the call


def test_gradients_flow_back_through_the_result():
    gen = torch.Generator().manual_seed(62)
    x = _real_values(gen, [3, 4]).requires_grad_()
    gain = _real_values(gen, [4]).requires_grad_()
    mix = _real_values(gen, [3, 4])
    inputs = (x, gain)
    try:
        out = rms_norm(x, gain, 1e-6)
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
    want = torch.autograd.grad((_oracle(x, gain, 1e-6) * mix).sum(), inputs, allow_unused=True)
    for g, w in zip(got, want, strict=True):
        expected = torch.zeros((), dtype=torch.float64) if w is None else w
        have = torch.zeros_like(expected) if g is None else g
        assert torch.allclose(have, expected, rtol=1e-7, atol=1e-9), (
            "a gradient through the result is wrong or missing: an input or a statistic was "
            "taken out of the autograd graph"
        )
