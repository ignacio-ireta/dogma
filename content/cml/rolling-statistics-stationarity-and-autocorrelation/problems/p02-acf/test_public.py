import numpy as np
import pytest
from submission import acf


def test_three_lags_match_hand_computed_values():
    result = acf([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], 3)
    np.testing.assert_allclose(
        result,
        [1.0, 0.1, 0.342857142857, -0.442857142857],
        rtol=1e-9,
        atol=1e-9,
        err_msg="autocorrelations differ: centre by the mean, divide by the total sum of squares",
    )


def test_lag_one_of_a_short_ramp_is_one_quarter():
    result = acf([1.0, 2.0, 3.0, 4.0], 1)
    assert result.shape == (2,), "nlags = 1 must return lags 0 and 1"
    assert result[0] == 1.0, "lag 0 is exactly one for any non-constant series"
    np.testing.assert_allclose(result[1], 0.25, rtol=1e-12, err_msg="lag 1 of 1..4 is 0.25")


def test_constant_series_raises_value_error():
    try:
        acf([2.0, 2.0, 2.0, 2.0], 1)
    except ValueError:
        return
    raise AssertionError("a constant series was accepted: its autocorrelation is undefined")


def test_nlags_must_be_below_the_length():
    try:
        acf([1.0, 2.0, 4.0], 3)
    except ValueError:
        return
    raise AssertionError("nlags equal to the length was accepted: nlags must be at most n - 1")


@pytest.mark.parametrize("scale", [1e200, 1e-200], ids=["large-scale", "small-scale"])
def test_rescaling_cannot_overflow_or_underflow_the_autocorrelation(scale):
    values = np.array([1.0, 2.0, 3.0]) * scale
    np.testing.assert_allclose(
        acf(values, 2),
        [1.0, 0.0, -0.5],
        rtol=1e-12,
        atol=1e-12,
        err_msg="ACF must stay scale-free when raw squared deviations overflow or underflow",
    )
