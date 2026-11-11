from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def rolling_mean_std(
    x: Sequence[float] | NDArray[np.float64],
    window: int,
    *,
    min_periods: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: trailing rolling mean and std, updated one value at a time.

    Position i uses only x[i - window + 1 .. i], so nothing after i is read (no look-ahead).
    Positions with fewer than ``min_periods`` values in the window are NaN. The standard
    deviation uses ddof = 1, so a window holding one value is NaN as well.

    The window is never recomputed from scratch. Welford's update adds the newest value and the
    reverse of the same update removes the value that left, so each step costs O(1).
    """
    values = _finite_1d(x)
    width = _positive_int(window, "window")
    need = (
        width
        if min_periods is None
        else (min_periods if min_periods == 0 else _positive_int(min_periods, "min_periods"))
    )
    if need > width:
        raise ValueError("min_periods must not exceed window")

    n = values.size
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    # Centre on the first value. The variance does not change when a constant is subtracted,
    # and the means are shifted back at the end. Without this, a level near 1e9 leaves the
    # running mean with about 1e-7 absolute resolution, and the error grows over a long series.
    shift = float(values[0]) if n else 0.0
    # Scale only by deviations seen so far, preserving the trailing (causal) contract.
    # Rescaling both the running mean and M2 keeps their units consistent, and prevents
    # large squared deviations from overflowing or tiny ones from underflowing.
    scale = 0.0
    count = 0
    mean = 0.0
    m2 = 0.0  # sum of squared deviations from the current window mean
    for i in range(n):
        # Add x[i]: the same update as one-pass Welford, with the count growing by one.
        raw_new = float(values[i])
        difference = raw_new - shift
        needed_scale = (
            abs(difference) if math.isfinite(difference) else max(abs(raw_new), abs(shift))
        )
        if needed_scale > scale:
            factor = scale / needed_scale
            mean *= factor
            m2 *= factor * factor
            scale = needed_scale
        new = _scaled_difference(raw_new, shift, scale)
        count += 1
        delta = new - mean
        mean += delta / count
        m2 += delta * (new - mean)

        # Remove x[i - width] once the window is full. Here `mean` is the mean of width + 1
        # values, and (mean - old) / (width) gives the change when one value leaves.
        if count > width:
            old = _scaled_difference(float(values[i - width]), shift, scale)
            count -= 1
            new_mean = mean + (mean - old) / count
            m2 -= (old - new_mean) * (old - mean)
            mean = new_mean

        if count >= need:
            means[i] = mean * scale + shift
            if count > 1:
                # Clamp at zero: rounding can leave m2 a few ulps below zero for constant data.
                stds[i] = math.sqrt(max(m2, 0.0) / (count - 1)) * scale
    return means, stds


def _scaled_difference(value: float, shift: float, scale: float) -> float:
    if scale == 0.0:
        return 0.0
    difference = value - shift
    if math.isfinite(difference):
        return difference / scale
    return value / scale - shift / scale


def _finite_1d(x: object) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers")
    return values


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1")
    return int(value)
