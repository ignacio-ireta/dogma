from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relu_family(
    x: ArrayLike, negative_slope: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: converts the input through float32 before computing.

    Real finite x; finite slope in [0, 1]. At zero the derivative convention is
    always 0. Invalid domain values raise ValueError. Inputs are not mutated.
    """
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    values = np.asarray(x).astype(np.float32).astype(np.float64)
    slope = float(negative_slope)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must be finite")
    if not np.isfinite(slope) or not 0 <= slope <= 1:
        raise ValueError("negative_slope must be finite and in [0, 1]")
    output = np.where(values > 0, values, slope * values)
    derivative = np.where(values > 0, 1.0, np.where(values < 0, slope, 0.0))
    return output, derivative
