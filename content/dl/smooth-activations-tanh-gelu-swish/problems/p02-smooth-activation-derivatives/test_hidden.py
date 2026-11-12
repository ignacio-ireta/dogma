from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit, ndtr
from submission import smooth_derivatives as _learner

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def smooth_derivatives(*args):
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
    s = expit(x)
    return (
        1 / np.cosh(x) ** 2,
        ndtr(x) + x * np.exp(-x * x / 2) / np.sqrt(2 * np.pi),
        s + x * s * expit(-x),
    )


def test_seeded_library_oracle():
    # Real pre-activations: standardised tumour measurements, doubled so the tails of the
    # activations are reached as well as the bend near zero.
    x, _ = sample_rows(
        "breast_cancer",
        data_rng("p02-smooth-activation-derivatives:oracle"),
        5,
        7,
        standardize=True,
    )
    x = 2.0 * x
    for got, want in zip(smooth_derivatives(x), _oracle(x), strict=True):
        _close(got, want, "activation formula disagrees with independent special-function oracle")


def test_scalar_and_integer_conversion():
    for got, want in zip(smooth_derivatives(2), _oracle(np.array(2.0)), strict=True):
        _close(got, want, "scalar input must return zero-dimensional float64 arrays")


def test_tails_do_not_overflow():
    x = np.array([-1000.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = smooth_derivatives(x)
    expected = ([0, 0], [0, 1], [0, 1])
    for got, want in zip(result, expected, strict=True):
        _close(got, want, "tail computation must use stable limiting behavior")


def test_invalid_domain_rejected():
    for x in ([np.nan], [np.inf], [-np.inf], [1001], [-1001], [1 + 2j]):
        _invalid(_learner, x)


def test_input_views_preserved():
    x = np.arange(-6.0, 6.0).reshape(3, 4)[:, ::2]
    before = x.copy()
    outputs = smooth_derivatives(x)
    assert np.array_equal(x, before), "activation must not mutate an input view"
    for value in outputs:
        value[...] = 7
    assert np.array_equal(x, before), "activation outputs must not alias the input"


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.floats(-6, 6, allow_nan=False, allow_infinity=False))
def test_elementwise_batch_consistency(value):
    batched = smooth_derivatives([value, value])
    scalar = smooth_derivatives(value)
    for got, want in zip(batched, scalar, strict=True):
        _close(got, np.repeat(want, 2), "activation must apply independently to every element")


def test_finite_difference_of_independent_forward_functions():
    x, _ = sample_rows(
        "breast_cancer", data_rng("p02-smooth-activation-derivatives:slope"), 4, 5, standardize=True
    )
    step = 1e-5

    def forward(v):
        return np.tanh(v), v * ndtr(v), v * expit(v)

    for analytic, high, low in zip(
        smooth_derivatives(x), forward(x + step), forward(x - step), strict=True
    ):
        numeric = (high - low) / (2 * step)
        assert np.allclose(analytic, numeric, atol=1e-9, rtol=1e-9), (
            "derivative fails independent central differences"
        )


def test_small_inputs_match_local_behavior():
    x = np.array([-1e-12, 1e-12])
    for got, want in zip(
        smooth_derivatives(x),
        (np.ones_like(x), 0.5 * np.ones_like(x), 0.5 * np.ones_like(x)),
        strict=True,
    ):
        assert np.allclose(got, want, atol=0, rtol=1e-9), (
            "small inputs must preserve the local linear behavior"
        )


def test_nonfinite_later_entries_rejected():
    for x in ([0.5, np.nan], [0.25, -0.5, np.inf], [1.0, 2.0, -np.inf]):
        _invalid(_learner, x)
