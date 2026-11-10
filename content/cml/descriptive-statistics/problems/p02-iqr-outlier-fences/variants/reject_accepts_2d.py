from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _quantile_linear(sorted_values: NDArray[np.float64], q: float) -> float:
    """Quantile of already sorted data by linear interpolation between order statistics.

    The position h = q * (n - 1) is fractional in general. The quantile is the value at floor(h)
    plus the fraction of the way to the next value. This is NumPy's default ('linear') method.
    """
    h = q * (sorted_values.shape[0] - 1)
    lo = math.floor(h)
    hi = min(lo + 1, sorted_values.shape[0] - 1)
    frac = h - lo
    return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))


def iqr_fences(xs: ArrayLike, k: float = 1.5) -> tuple[float, float]:
    """Wrong on purpose: flattens 2-D input instead of rejecting it. Tukey fences: (Q1 - k * IQR, Q3 + k * IQR) with IQR = Q3 - Q1 by linear interpolation.

    Values outside the fences are flagged as outliers. The quartiles move only when the values
    near them move, so one extreme value does not shift the fences the way it shifts the mean.
    """  # noqa: E501
    # np.array copies, so sorting below never touches the caller's data.
    values = np.array(xs, dtype=np.float64).ravel()
    if values.ndim != 1:
        raise ValueError(f"xs must be 1-D, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("need at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError("xs must contain only finite numbers")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    sorted_values = np.sort(values)
    q1 = _quantile_linear(sorted_values, 0.25)
    q3 = _quantile_linear(sorted_values, 0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr
