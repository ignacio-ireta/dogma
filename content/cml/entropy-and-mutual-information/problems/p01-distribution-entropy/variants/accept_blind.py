from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def entropy(p: ArrayLike, base: float = 2.0) -> float:
    """Return H(p) = -sum_i p_i * log_base(p_i), with 0 * log 0 taken as 0."""
    base = float(base)
    if not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("base must be finite, greater than 0, and not equal to 1")
    arr = np.array(p, dtype=float, copy=True)
    if arr.ndim != 1:
        raise ValueError("p must be one-dimensional")
    if arr.size == 0:
        raise ValueError("p must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("p must contain only finite entries")
    if np.any(arr < 0):
        raise ValueError("p must contain only non-negative entries")
    if abs(float(arr.sum()) - 1.0) > 1e-9:
        raise ValueError("p must sum to 1 within 1e-9")
    nonzero = arr[arr > 0]
    total = -float(np.sum(nonzero * np.log(nonzero))) / math.log(base)
    return float(total) + 0.0
