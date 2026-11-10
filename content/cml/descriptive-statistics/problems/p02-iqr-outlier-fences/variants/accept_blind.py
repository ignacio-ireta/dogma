from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _interpolated_quantile(sorted_values: np.ndarray, q: float) -> float:
    n = sorted_values.shape[0]
    h = q * (n - 1)
    i = math.floor(h)
    j = min(i + 1, n - 1)
    return float(sorted_values[i] + (h - i) * (sorted_values[j] - sorted_values[i]))


def iqr_fences(xs: ArrayLike, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences (Q1 - k * IQR, Q3 + k * IQR) from linearly interpolated quartiles."""
    arr = np.asarray(xs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("xs must be 1-D")
    if arr.size == 0:
        raise ValueError("xs must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError("xs must contain only finite values")
    k = float(k)
    if not k >= 0:
        raise ValueError("k must be non-negative")
    s = np.sort(arr)  # sorted copy; the caller's array is untouched
    q1 = _interpolated_quantile(s, 0.25)
    q3 = _interpolated_quantile(s, 0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)
