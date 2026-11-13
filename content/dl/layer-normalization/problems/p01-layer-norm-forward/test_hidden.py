# Hidden tests: oracle comparisons against PyTorch's functional layer norm, plus edge cases.
# The oracle is used only here; the submission may not use torch.nn or torch.layer_norm.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import layer_norm

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
    key = f"p01-layer-norm-forward:values:{draw}"
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
        "breast_cancer", rng(f"layer-norm-p01:features:{seed}"), rows, shape[-1], standardize=True
    )
    return torch.tensor(drawn.reshape(shape), dtype=torch.float64).to(dtype) * scale + offset


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return _real_values(g, [shape], dtype) * scale + offset


def _call(x, weight, bias, eps=1e-5):
    snapshot = [x.clone(), weight.clone(), bias.clone()]
    try:
        out = layer_norm(x, weight, bias, eps=eps)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, dtypes and the contract",
            pytrace=False,
        )
    for name, before, after in zip(
        ("x", "weight", "bias"), snapshot, (x, weight, bias), strict=True
    ):
        if not torch.equal(before, after):
            pytest.fail(f"input {name} was modified in place; return a new tensor", pytrace=False)
    if not torch.is_tensor(out):
        pytest.fail("return a torch tensor, not a NumPy array or a list", pytrace=False)
    if out.dtype != x.dtype:
        pytest.fail("output must keep the dtype of x; do not downcast", pytrace=False)
    return out


def _oracle(x, weight, bias, eps=1e-5):
    return ref_f.layer_norm(x, (x.shape[-1],), weight, bias, eps)


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
        weight, bias = _randn((d,), seed + 10), _randn((d,), seed + 20)
        y = _call(x, weight, bias, eps=1e-5)
        _close(
            y,
            _oracle(x, weight, bias),
            "output differs from the reference layer norm: check that statistics run over the last "
            "dimension (per example), the biased variance (divide by D), eps inside the square "
            "root, and that the mean is subtracted before scaling",
        )


def test_each_example_is_normalized_independently_of_the_others():
    x = _features((5, 4), 31, scale=3.0, offset=-1.0)
    weight, bias = _randn((4,), 32), _randn((4,), 33)
    together = _call(x, weight, bias)
    for i in range(x.shape[0]):
        alone = _call(x[i : i + 1], weight, bias)
        _close(
            together[i : i + 1],
            alone,
            "an example's output depends on other examples: statistics leaked across the batch",
        )


def test_statistics_run_over_the_last_axis_not_the_batch():
    # Rows and features differ, so reducing over the batch cannot match by accident.
    x = _features((3, 5), 41, scale=2.0, offset=1.0)
    weight = torch.ones(5, dtype=torch.float64)
    bias = torch.zeros(5, dtype=torch.float64)
    y = _call(x, weight, bias, eps=1e-12)
    _close(
        y,
        _oracle(x, weight, bias, eps=1e-12),
        "normalized over the wrong axis: statistics are per example, over the last dimension",
    )


def test_rows_are_centred_before_scaling():
    x = _features((4, 6), 51, scale=5.0, offset=3.0)
    weight = torch.ones(6, dtype=torch.float64)
    bias = torch.zeros(6, dtype=torch.float64)
    y = _call(x, weight, bias)
    if not torch.allclose(y.mean(dim=-1), torch.zeros(4, dtype=torch.float64), atol=1e-8):
        pytest.fail(
            "output rows are not centred: subtract the mean over the last dimension", pytrace=False
        )


def test_eps_is_added_inside_the_square_root():
    # Variance here is 2.5e-7 and eps is 1e-6: the placement changes the output by a large factor.
    x = torch.tensor([[0.0, 1e-3]], dtype=torch.float64)
    weight = torch.ones(2, dtype=torch.float64)
    bias = torch.zeros(2, dtype=torch.float64)
    y = _call(x, weight, bias, eps=1e-6)
    _close(
        y,
        _oracle(x, weight, bias, eps=1e-6),
        "eps is added outside the square root; add it to the variance before taking sqrt",
    )


def test_variance_is_biased_divide_by_feature_count():
    x = _features((4, 3), 61)
    weight = torch.ones(3, dtype=torch.float64)
    bias = torch.zeros(3, dtype=torch.float64)
    y = _call(x, weight, bias, eps=1e-12)
    centred = x - x.mean(dim=-1, keepdim=True)
    unbiased = centred / torch.sqrt(centred.pow(2).sum(dim=-1, keepdim=True) / 2 + 1e-12)
    if torch.allclose(y, unbiased, atol=1e-6):
        pytest.fail(
            "used unbiased variance (divide by D - 1); layer norm divides by D", pytrace=False
        )
    _close(y, _oracle(x, weight, bias, eps=1e-12), "output differs from the reference layer norm")


def test_large_common_offset_keeps_statistics_accurate():
    # E[x^2] - mean^2 loses most of its digits when every feature sits near 1e6.
    for seed, offset in enumerate((1e6, 1e8)):
        x = _features((3, 4), 71 + seed, scale=1.0, offset=offset)
        weight, bias = _randn((4,), 73), _randn((4,), 74)
        y = _call(x, weight, bias)
        _close(
            y,
            _oracle(x, weight, bias),
            "large common offset destroyed the statistics: centre the inputs before squaring",
            atol=1e-7,
            rtol=1e-7,
        )


def test_tiny_magnitudes_match_reference():
    x = _features((4, 3), 81, scale=1e-8)
    weight, bias = _randn((3,), 82), _randn((3,), 83)
    y = _call(x, weight, bias, eps=1e-12)
    _close(
        y, _oracle(x, weight, bias, eps=1e-12), "tiny-magnitude output differs from the reference"
    )


def test_constant_example_returns_bias():
    x = torch.full((2, 4), 7.5, dtype=torch.float64)
    weight = _randn((4,), 91)
    bias = _randn((4,), 92)
    y = _call(x, weight, bias)
    _close(
        y, bias.expand(2, 4), "a constant example must return the bias: the centred values are zero"
    )


def test_single_feature_returns_bias():
    x = _features((3, 1), 101)
    weight, bias = _randn((1,), 102), _randn((1,), 103)
    y = _call(x, weight, bias)
    _close(y, bias.expand(3, 1), "a single feature must return the bias")


def test_float32_inputs_return_float32():
    x = _features((3, 5), 111, dtype=torch.float32)
    weight = torch.ones(5, dtype=torch.float32)
    bias = torch.zeros(5, dtype=torch.float32)
    y = _call(x, weight, bias)
    assert y.dtype == torch.float32, "output dtype must follow the input dtype"
    want = _oracle(x, weight, bias)
    _close(
        y,
        want,
        "float32 output differs from the reference beyond float32 precision",
        atol=1e-5,
        rtol=1e-5,
    )


def test_empty_leading_dimension_returns_empty_output():
    x = torch.empty((0, 4), dtype=torch.float64)
    y = _call(x, torch.ones(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
    assert tuple(y.shape) == (0, 4), "an empty batch must return an empty output of the same shape"


def test_invalid_arguments_raise_value_error():
    good_w = torch.ones(3, dtype=torch.float64)
    good_b = torch.zeros(3, dtype=torch.float64)
    x = _features((2, 3), 121)
    cases = {
        "a 0-dimensional x must raise ValueError": lambda: layer_norm(
            torch.tensor(1.0), good_w, good_b
        ),
        "an integer x must raise ValueError": lambda: layer_norm(
            torch.ones(2, 3, dtype=torch.int64), good_w, good_b
        ),
        "a weight with the wrong length must raise ValueError": lambda: layer_norm(
            x, torch.ones(2, dtype=torch.float64), good_b
        ),
        "a bias with the wrong length must raise ValueError": lambda: layer_norm(
            x, good_w, torch.zeros(4, dtype=torch.float64)
        ),
        "a zero-width last dimension must raise ValueError": lambda: layer_norm(
            torch.empty(2, 0, dtype=torch.float64),
            torch.ones(0, dtype=torch.float64),
            torch.zeros(0, dtype=torch.float64),
        ),
        "a zero eps must raise ValueError": lambda: layer_norm(x, good_w, good_b, eps=0.0),
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


def test_one_dimensional_single_feature_keeps_shape():
    # A length-one vector is constant, so its centred value is zero and the output is the bias.
    # The output must keep the input's shape (1,), not collapse to a scalar.
    x = _features((1,), 111)
    weight, bias = _randn((1,), 112), _randn((1,), 113)
    y = _call(x, weight, bias)
    _close(y, bias, "a one-dimensional input must return a tensor of the same shape")


def test_non_contiguous_three_dimensional_input_is_normalized():
    # A transposed tensor is a valid float input of shape (2, 3, 5) with non-contiguous memory.
    base = _features((5, 3, 2), 121)
    x = base.transpose(0, 2)
    weight, bias = _randn((5,), 122), _randn((5,), 123)
    y = _call(x, weight, bias)
    _close(
        y,
        _oracle(x, weight, bias),
        "a valid non-contiguous input must be normalized over its last dimension like any other",
    )


def test_gradients_flow_back_through_the_result():
    gen = torch.Generator().manual_seed(61)
    x = _real_values(gen, [3, 4]).requires_grad_()
    weight = _real_values(gen, [4]).requires_grad_()
    bias = _real_values(gen, [4]).requires_grad_()
    mix = _real_values(gen, [3, 4])
    inputs = (x, weight, bias)
    try:
        out = layer_norm(x, weight, bias, 1e-5)
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
        (ref_f.layer_norm(x, (4,), weight, bias, 1e-5) * mix).sum(), inputs, allow_unused=True
    )
    for g, w in zip(got, want, strict=True):
        expected = torch.zeros((), dtype=torch.float64) if w is None else w
        have = torch.zeros_like(expected) if g is None else g
        assert torch.allclose(have, expected, rtol=1e-7, atol=1e-9), (
            "a gradient through the result is wrong or missing: an input or a statistic was "
            "taken out of the autograd graph"
        )
