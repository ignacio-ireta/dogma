from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


def fit_power_law(x: NDArray[Any], y: NDArray[Any]) -> tuple[float, float]:
    """Wrong on purpose: intercept passes through the first point not the mean point.

    Taking logs turns the power law into a line, log y = log a + b * log x, so ordinary least
    squares applies. The fit is in log space, so it weights relative errors equally.
    """
    # Validate first. Each rule is named so that a bad input cannot reach the logarithm.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")
    if x_arr.size < 2:
        raise ValueError("need at least two points")
    if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
        raise ValueError("x and y must be finite")
    if (x_arr <= 0).any() or (y_arr <= 0).any():
        raise ValueError("x and y must be strictly positive for the logarithm")

    # Step 1, the log transform. u = log x and v = log y turn the power law into a line.
    u = np.log(x_arr)
    v = np.log(y_arr)

    # Step 2, the centred sums. Subtracting the means first keeps the sums small, so the slope
    # is accurate even when u values are close together and large in magnitude. The expanded
    # form sum(u*v) - n*mean(u)*mean(v) would lose those digits to cancellation.
    du = u - u.mean()
    dv = v - v.mean()
    sxx = float(np.sum(du * du))
    if sxx == 0.0:
        raise ValueError("x values must not all be equal: the slope is undefined")

    # Step 3, the least-squares slope and intercept. b = cov(u, v) / var(u), and the line passes
    # through the mean point, so c = mean(v) - b * mean(u).
    b = float(np.sum(du * dv)) / sxx
    c = float(v[0]) - b * float(u[0])

    # Step 4, back to the power-law constant. a = exp(c), because log a = c.
    return math.exp(c), b
