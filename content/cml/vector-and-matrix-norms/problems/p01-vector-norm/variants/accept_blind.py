from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def vector_norm(x: ArrayLike, p: float) -> float:
    """Return the p-norm of a 1-D vector x, for real p >= 1 or p = inf.

    Parameters
    ----------
    x : 1-D array-like of finite real numbers, length n >= 1 (ints allowed).
    p : real number >= 1, or ``np.inf``.

    Returns
    -------
    float >= 0: (sum_i |x_i|**p) ** (1/p), or max_i |x_i| when p is infinite.
    Must not overflow or underflow for entries anywhere in the float64 range.

    Raises
    ------
    ValueError
        If p is below 1 or NaN, if x is not 1-D, or if x is empty.
    """
    if p is None or math.isnan(float(p)):
        raise ValueError("p must be a number at least 1 (or infinity); got NaN")
    p = float(p)
    if p < 1:
        raise ValueError(f"p must be at least 1 (or infinity); got {p}")

    arr = np.array(x, dtype=np.float64)  # copy, so the input is never mutated
    if arr.ndim != 1:
        raise ValueError(f"expected a one-dimensional vector, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("expected a non-empty vector")

    absx = np.abs(arr)
    scale = float(np.max(absx))
    if scale == 0.0:
        return 0.0
    if math.isinf(p):
        return scale

    # Scale by the largest entry so that every |y_i| <= 1: no overflow in the powers,
    # and the largest term is exactly 1 so the sum cannot underflow to zero.
    y = absx / scale
    total = float(np.sum(y**p))
    return float(scale * total ** (1.0 / p))
