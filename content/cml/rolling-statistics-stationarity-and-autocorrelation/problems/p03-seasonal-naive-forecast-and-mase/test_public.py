import math

import numpy as np
from submission import seasonal_naive_forecast_and_mase


def test_season_three_forecast_and_mase_match_hand_values():
    forecast, mase = seasonal_naive_forecast_and_mase(
        [10.0, 22.0, 31.0, 12.0, 19.0, 33.0, 9.0, 21.0], [20.0, 25.0, 11.0, 30.0], 3
    )
    np.testing.assert_allclose(
        forecast,
        [33.0, 9.0, 21.0, 33.0],
        rtol=1e-12,
        err_msg="forecast must repeat the last complete season of train",
    )
    assert abs(mase - 4.375) < 1e-12, (
        "MASE must be the test error (mean 10.5) over the in-sample seasonal-naive scale (mean 2.4)"
    )


def test_season_one_is_the_naive_forecast_with_a_lag_one_scale():
    forecast, mase = seasonal_naive_forecast_and_mase([1.0, 2.0, 3.0, 4.0], [5.0, 6.0], 1)
    np.testing.assert_allclose(
        forecast, [4.0, 4.0], rtol=1e-12, err_msg="naive forecast repeats the last value"
    )
    assert abs(mase - 1.5) < 1e-12, (
        "MASE with season 1 scales by the mean lag-one difference (here 1)"
    )


def test_zero_scale_training_series_raises_value_error():
    try:
        seasonal_naive_forecast_and_mase([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [1.0], 3)
    except ValueError:
        return
    raise AssertionError(
        "a perfectly periodic training series was accepted: the MASE scale is zero"
    )


def test_empty_test_raises_value_error():
    try:
        seasonal_naive_forecast_and_mase([1.0, 2.0, 3.0, 4.0], [], 2)
    except ValueError:
        return
    raise AssertionError("an empty test was accepted: at least one forecast step is required")


def test_mase_remains_scale_free_when_the_unscaled_error_sum_would_overflow():
    scale = 1e308
    forecast, mase = seasonal_naive_forecast_and_mase(
        [-scale, scale, -scale, scale], [-scale, scale], 1
    )
    np.testing.assert_array_equal(forecast, [scale, scale])
    assert math.isclose(mase, 0.5, rel_tol=1e-12), "MASE must scale errors before reducing them"
