from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def rolling_mean_std(
    x: Sequence[float] | NDArray[np.float64], window: int, *, min_periods: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trailing rolling mean and sample standard deviation, updated one value at a time.

    Position i uses only x[i - window + 1 .. i], so nothing after i is read (no look-ahead).
    Positions with fewer than ``min_periods`` values in the window are NaN. The standard
    deviation uses ddof = 1, so a window holding one value is NaN as well.

    The window is never recomputed from scratch. Welford's update adds the newest value and the
    reverse of the same update removes the value that left, so each step costs O(1).
    """
    values = _finite_1d(x)
    width = _positive_int(window, "window")
    need = width if min_periods is None else _positive_int(min_periods, "min_periods")
    if need > width:
        raise ValueError("min_periods must not exceed window")
    n = values.size
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    shift = float(values[0]) if n else 0.0
    scale = 0.0
    count = 0
    mean = 0.0
    m2 = 0.0
    for i in range(n):
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
        if count > width:
            old = _scaled_difference(float(values[i - width]), shift, scale)
            count -= 1
            new_mean = mean + (mean - old) / count
            m2 -= (old - new_mean) * (old - mean)
            mean = new_mean
        if count >= need:
            means[i] = mean * scale + shift
            if count > 1:
                stds[i] = math.sqrt(max(m2, 0.0) / (count - 1)) * scale
    return (means, stds)


def _scaled_difference(value: float, shift: float, scale: float) -> float:
    if scale == 0.0:
        return 0.0
    difference = value - shift
    if math.isfinite(difference):
        return difference / scale
    return value / scale - shift * scale


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
