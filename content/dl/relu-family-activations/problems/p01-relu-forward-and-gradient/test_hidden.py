from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import relu_family as _learner

from dogma_data import rng as data_rng
from dogma_data import sample_rows


def relu_family(*args):
    try:
        result = _learner(*args)
    except Exception as exc:
        pytest.fail(f"valid input violated the computation contract: {type(exc).__name__}")
    assert isinstance(result, tuple) and len(result) == 2, (
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


def test_seeded_scalar_piecewise_oracle():
    # Real inputs of both signs: five tumours by seven standardised breast_cancer measurements.
    x, _ = sample_rows("breast_cancer", data_rng("relu-family-p01:oracle"), 5, 7, standardize=True)
    for slope in (0.0, 0.03, 0.8, 1.0):
        y, d = relu_family(x, slope)
        expected = np.array([v if v > 0 else slope * v for v in x.flat]).reshape(x.shape)
        deriv = np.array([1.0 if v > 0 else slope for v in x.flat]).reshape(x.shape)
        _close(y, expected, "piecewise values disagree with scalar oracle")
        _close(d, deriv, "local derivatives disagree with scalar oracle")


def test_zero_convention_including_identity_slope():
    for slope in (0.0, 0.2, 1.0):
        y, d = relu_family(0, slope)
        _close(y, np.array(0.0), "scalar value must be a zero-dimensional array")
        _close(d, np.array(0.0), "derivative at zero must use the explicit convention")


def test_finite_difference_away_from_kink():
    x = np.array([-4.0, -0.3, 0.2, 5.0])
    step = 1e-6
    for slope in (0.0, 0.1, 1.0):
        _, d = relu_family(x, slope)

        def independent_forward(v, leak=slope):
            return np.maximum(v, 0) + leak * np.minimum(v, 0)

        numeric = (independent_forward(x + step) - independent_forward(x - step)) / (2 * step)
        assert np.allclose(d, numeric, atol=1e-9, rtol=1e-9), (
            "local derivative fails finite differences"
        )


def test_invalid_nonfinite_and_slope_rules():
    for x in ([np.nan], [np.inf], [complex(1, 2)]):
        _invalid(_learner, x)
    for slope in (-0.1, 1.1, np.nan, np.inf):
        _invalid(_learner, [1], slope)


def test_extreme_magnitudes_and_integer_conversion():
    x = np.array([-1e200, -1e-200, 1e-200, 1e200])
    y, d = relu_family(x, 0.25)
    _close(y, [-2.5e199, -2.5e-201, 1e-200, 1e200], "finite extremes must retain branch scale")
    _close(d, [0.25, 0.25, 1, 1], "magnitude must not change the derivative mask")
    _close(relu_family([1, -2])[0], [1, 0], "integer inputs must convert to float64")


def test_input_views_not_mutated_or_aliased():
    x = np.arange(-8.0, 8.0).reshape(4, 4)[:, ::2]
    before = x.copy()
    y, d = relu_family(x, 0.2)
    assert np.array_equal(x, before), "input view was mutated"
    y[...] = 9
    d[...] = 8
    assert np.array_equal(x, before), "output must not alias input storage"


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.floats(-10, 10, allow_nan=False, allow_infinity=False), st.floats(0, 1))
def test_positive_rescaling_preserves_derivative(value, slope):
    y, d = relu_family([value], slope)
    y2, d2 = relu_family([3 * value], slope)
    _close(y2, 3 * y, "positive homogeneity must hold")
    _close(d2, d, "positive scaling must preserve the sign region")


def test_small_negative_values_are_not_clipped_for_a_leak():
    x = np.array([-1e-200, 1e-200])
    y, _ = relu_family(x, 0.25)
    assert np.allclose(y, [-2.5e-201, 1e-200], atol=0, rtol=1e-12), (
        "tiny finite values must retain their branch scale"
    )
