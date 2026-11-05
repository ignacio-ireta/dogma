# Hidden tests: oracle comparisons against PyTorch's functional layer norm, plus edge cases.
# The oracle is used only here; the submission may not use torch.nn or torch.layer_norm.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import layer_norm

ATOL = 1e-9
RTOL = 1e-9


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=dtype) * scale + offset


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
        x = _randn(shape, seed, scale=2.0, offset=0.5)
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
    x = _randn((5, 4), 31, scale=3.0, offset=-1.0)
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
    x = _randn((3, 5), 41, scale=2.0, offset=1.0)
    weight = torch.ones(5, dtype=torch.float64)
    bias = torch.zeros(5, dtype=torch.float64)
    y = _call(x, weight, bias, eps=1e-12)
    _close(
        y,
        _oracle(x, weight, bias, eps=1e-12),
        "normalized over the wrong axis: statistics are per example, over the last dimension",
    )


def test_rows_are_centred_before_scaling():
    x = _randn((4, 6), 51, scale=5.0, offset=3.0)
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
    x = _randn((4, 3), 61)
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
        x = _randn((3, 4), 71 + seed, scale=1.0, offset=offset)
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
    x = _randn((4, 3), 81, scale=1e-8)
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
    x = _randn((3, 1), 101)
    weight, bias = _randn((1,), 102), _randn((1,), 103)
    y = _call(x, weight, bias)
    _close(y, bias.expand(3, 1), "a single feature must return the bias")


def test_float32_inputs_return_float32():
    x = _randn((3, 5), 111, dtype=torch.float32)
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
    x = _randn((2, 3), 121)
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
