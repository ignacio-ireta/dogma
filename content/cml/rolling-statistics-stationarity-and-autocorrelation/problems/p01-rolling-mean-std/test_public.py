import math

import numpy as np
from submission import rolling_mean_std


def test_window_three_matches_hand_values_with_default_min_periods():
    means, stds = rolling_mean_std([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert np.isnan(means[:2]).all() and np.isnan(stds[:2]).all(), (
        "positions with fewer than window values must be NaN when min_periods is not given"
    )
    np.testing.assert_allclose(
        means[2:], [2.0, 3.0, 4.0], rtol=1e-12, atol=1e-12, err_msg="rolling mean is wrong"
    )
    np.testing.assert_allclose(
        stds[2:], [1.0, 1.0, 1.0], rtol=1e-12, atol=1e-12, err_msg="rolling std is wrong"
    )


def test_min_periods_one_keeps_partial_windows_and_nan_spread_for_one_value():
    means, stds = rolling_mean_std([1.0, 2.0, 3.0, 4.0, 5.0], 3, min_periods=1)
    np.testing.assert_allclose(
        means, [1.0, 1.5, 2.0, 3.0, 4.0], rtol=1e-12, atol=1e-12, err_msg="rolling mean is wrong"
    )
    assert math.isnan(stds[0]), "a window holding one value has no sample standard deviation"
    np.testing.assert_allclose(
        stds[1:],
        [math.sqrt(0.5), 1.0, 1.0, 1.0],
        rtol=1e-12,
        err_msg="the sample standard deviation divides by n - 1",
    )


def test_window_longer_than_series_is_all_nan_by_default():
    means, stds = rolling_mean_std([2.0, 4.0], 5)
    assert means.shape == (2,) and stds.shape == (2,), "output length must match the input length"
    assert np.isnan(means).all() and np.isnan(stds).all(), (
        "no window reaches min_periods, so every entry must be NaN"
    )


def test_empty_input_gives_empty_outputs():
    means, stds = rolling_mean_std([], 3)
    assert means.shape == (0,) and stds.shape == (0,), "empty input must give empty outputs"


def test_window_zero_raises_value_error():
    try:
        rolling_mean_std([1.0, 2.0], 0)
    except ValueError:
        return
    raise AssertionError("a window of zero was accepted: window must be an integer of at least 1")


def test_large_and_small_readings_keep_a_representable_sample_spread():
    for scale in (1e200, 1e-200):
        means, stds = rolling_mean_std([-scale, scale, -scale], 2)
        np.testing.assert_allclose(
            means[1:] / scale,
            [0.0, 0.0],
            atol=1e-12,
            err_msg="a symmetric pair has a mean of zero up to rounding, relative to the scale",
        )
        np.testing.assert_allclose(
            stds[1:] / scale,
            [np.sqrt(2.0), np.sqrt(2.0)],
            rtol=1e-12,
            atol=0.0,
            err_msg="sample spread must survive large and small finite magnitudes",
        )
