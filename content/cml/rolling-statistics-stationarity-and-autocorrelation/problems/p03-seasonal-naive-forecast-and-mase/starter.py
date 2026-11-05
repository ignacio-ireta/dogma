from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def seasonal_naive_forecast_and_mase(
    train: Sequence[float] | NDArray[np.float64],
    test: Sequence[float] | NDArray[np.float64],
    season: int,
) -> tuple[NDArray[np.float64], float]:
    """Seasonal-naive forecast of len(test) steps after ``train``, and the MASE of that forecast.

    Forecast: yhat(n + h) = y(n + h - season * ceil(h / season)), with n = len(train), h = 1, 2, ...
    Scale: mean of |y_t - y_(t - season)| over t = season + 1 .. n, using the training data only.
    MASE: mean absolute error of the forecast against ``test``, divided by the scale.

    Returns
    -------
    (forecast, mase) : forecast is a (len(test),) float64 array; mase is a float.

    Raises
    ------
    ValueError
        If an input is not finite and one-dimensional, test is empty, season is not a positive
        integer, len(train) < season + 1, or the scale is zero.
    """
    raise NotImplementedError
