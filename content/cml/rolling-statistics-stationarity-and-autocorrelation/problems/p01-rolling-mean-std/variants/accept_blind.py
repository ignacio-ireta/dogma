import math
import numbers

import numpy as np


def rolling_mean_std(x, window, *, min_periods=None):
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers")
    for name, value in (("window", window), ("min_periods", min_periods)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1
        ):
            raise ValueError(f"{name} must be an integer of at least 1")
    width = int(window)
    need = width if min_periods is None else int(min_periods)
    if need > width:
        raise ValueError("min_periods must not exceed window")
    n = values.size
    # scale by the largest magnitude so squares cannot overflow or underflow; scale back at the end
    scale = float(np.max(np.abs(values))) if n else 1.0
    if scale == 0.0:
        scale = 1.0
    scaled = values / scale
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    for i in range(n):
        # trailing window: positions i - width + 1 .. i, never a later value
        segment = scaled[max(0, i - width + 1) : i + 1]
        if segment.size >= need:
            means[i] = segment.mean() * scale
            if segment.size > 1:
                stds[i] = (
                    math.sqrt(float(np.sum((segment - segment.mean()) ** 2)) / (segment.size - 1))
                    * scale
                )
    return means, stds
