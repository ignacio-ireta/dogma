from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relu_family(
    x: ArrayLike, negative_slope: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return leaky-ReLU values and local derivatives with the same shape as x.

    Real finite array-like x is converted to float64; slope must be finite in [0, 1].
    At x == 0 the derivative convention is 0, for every slope. Invalid values raise
    ValueError. Scalars and empty arrays are accepted. Inputs are not mutated."""
    if isinstance(negative_slope, bool) or not isinstance(negative_slope, numbers.Real):
        raise ValueError("negative_slope must be a real scalar")
    slope = float(negative_slope)
    if not np.isfinite(slope) or slope < 0.0 or slope > 1.0:
        raise ValueError("negative_slope must be finite and in [0, 1]")

    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError("x must be real")
    try:
        xs = np.array(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be real numeric") from exc
    if not np.all(np.isfinite(xs)):
        raise ValueError("x must be finite")

    values = np.where(xs > 0, xs, slope * xs).astype(np.float64, copy=True)
    derivatives = np.where(xs > 0, 1.0, np.where(xs < 0, slope, 0.0)).astype(np.float64, copy=True)
    return values, derivatives
