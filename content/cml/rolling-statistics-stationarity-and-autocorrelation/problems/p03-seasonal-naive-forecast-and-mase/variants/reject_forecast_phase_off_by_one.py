from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def seasonal_naive_forecast_and_mase(
    train: Sequence[float] | NDArray[np.float64],
    test: Sequence[float] | NDArray[np.float64],
    season: int,
) -> tuple[NDArray[np.float64], float]:
    """Wrong on purpose: seasonal-naive forecast of len(test) steps, and its MASE.

    Forecast: yhat(n + h) = y(n + h - season * ceil(h / season)), where n = len(train) and
    h = 1, 2, ... The forecast repeats the last observed season. With season = 1 it is the naive
    forecast, which repeats the last value.

    MASE divides the mean absolute test error by the in-sample mean absolute error of the same
    seasonal-naive rule on the training data:

        scale = mean over t = season .. n-1 of |y_t - y_{t - season}|
        MASE  = mean over the test steps of |actual - forecast| / scale

    Returns
    -------
    (forecast, mase) : forecast is a (len(test),) float64 array; mase is a float.
    """
    y = _finite_1d(train, "train")
    actual = _finite_1d(test, "test")
    m = _positive_int(season, "season")
    n = y.size
    if n < m + 1:
        raise ValueError("train needs at least season + 1 values for the seasonal scale")
    if actual.size == 0:
        raise ValueError("test must contain at least one value")

    # Step h (1-based) reads the same phase of the last complete season. Using the modulus of
    # (h - 1) picks that phase without a loop over full seasons.
    steps = np.arange(1, actual.size + 1)
    forecast = y[n - m + (steps % m)].astype(np.float64)

    # The scale uses the training series only, never the test values (no leakage).
    # Work in common scaled units before differences and sums. The common factor cancels
    # in MASE, while raw errors such as abs(-1e308 - 1e308) would overflow.
    magnitude = float(np.max(np.abs(y)))
    if magnitude == 0.0:
        raise ValueError("the training series repeats exactly, so the MASE scale is zero")
    scaled_y = y / magnitude
    scaled_actual = actual / magnitude
    scaled_forecast = forecast / magnitude
    in_sample_errors = np.abs(scaled_y[m:] - scaled_y[: n - m])  # n - m terms
    scale = float(np.mean(in_sample_errors))
    if scale == 0.0:
        raise ValueError("the training series repeats exactly, so the MASE scale is zero")
    mase = float(np.mean(np.abs(scaled_actual - scaled_forecast)) / scale)
    return forecast, mase


def _finite_1d(x: object, name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite numbers")
    return values


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1")
    return int(value)
