from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit, ndtr
from submission import smooth_activations as _learner


def smooth_activations(*args):
    try:
        result = _learner(*args)
    except Exception as exc:
        pytest.fail(f"valid input violated the computation contract: {type(exc).__name__}")
    assert isinstance(result, tuple) and len(result) == 3, (
        "return tuple length violates the contract"
    )
    for value in result:
        assert isinstance(value, np.ndarray), "return values must be NumPy arrays"
        assert value.dtype == np.float64, "output dtype must be float64"
        assert value.shape == np.asarray(args[0]).shape, "output shape must preserve input axes"
    return result


def _close(got, want, message):
    assert isinstance(got, np.ndarray), "return type must be a NumPy array"
    assert got.dtype == np.float64, "output dtype must be float64"
    assert got.shape == np.shape(want), "output shape must match the contract"
    assert np.allclose(got, want, atol=1e-9, rtol=1e-9), message


def _invalid(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid input was accepted instead of raising ValueError")


def _oracle(x):
    return np.tanh(x), x * ndtr(x), x * expit(x)


def test_seeded_library_oracle():
    x = np.random.default_rng(19).uniform(-8, 8, (5, 7))
    for got, want in zip(smooth_activations(x), _oracle(x), strict=True):
        _close(got, want, "activation formula disagrees with independent special-function oracle")


def test_scalar_and_integer_conversion():
    for got, want in zip(smooth_activations(2), _oracle(np.array(2.0)), strict=True):
        _close(got, want, "scalar input must return zero-dimensional float64 arrays")


def test_tails_do_not_overflow():
    x = np.array([-1000.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = smooth_activations(x)
    expected = ([-1, 1], [0, 1000], [0, 1000])
    for got, want in zip(result, expected, strict=True):
        _close(got, want, "tail computation must use stable limiting behavior")


def test_invalid_domain_rejected():
    for x in ([np.nan], [np.inf], [-np.inf], [1001], [-1001], [1 + 2j]):
        _invalid(_learner, x)


def test_input_views_preserved():
    x = np.arange(-6.0, 6.0).reshape(3, 4)[:, ::2]
    before = x.copy()
    outputs = smooth_activations(x)
    assert np.array_equal(x, before), "activation must not mutate an input view"
    for value in outputs:
        value[...] = 7
    assert np.array_equal(x, before), "activation outputs must not alias the input"


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.floats(-6, 6, allow_nan=False, allow_infinity=False))
def test_elementwise_batch_consistency(value):
    batched = smooth_activations([value, value])
    scalar = smooth_activations(value)
    for got, want in zip(batched, scalar, strict=True):
        _close(got, np.repeat(want, 2), "activation must apply independently to every element")


def test_gelu_is_exact_cdf_form_not_tanh_approximation():
    x = np.array([-3.0, -1.7, 0.7, 2.5])
    _close(smooth_activations(x)[1], x * ndtr(x), "GELU must use the specified exact CDF form")


def test_small_inputs_match_local_behavior():
    x = np.array([-1e-12, 1e-12])
    for got, want in zip(smooth_activations(x), (x, 0.5 * x, 0.5 * x), strict=True):
        assert np.allclose(got, want, atol=0, rtol=1e-9), (
            "small inputs must preserve the local linear behavior"
        )
