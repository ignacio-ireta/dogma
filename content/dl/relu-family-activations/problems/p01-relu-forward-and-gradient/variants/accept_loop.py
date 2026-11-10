from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relu_family(
    x: ArrayLike, negative_slope: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return same-shape float64 leaky-ReLU values and local derivatives.

    Real finite x; finite slope in [0, 1]. At zero the derivative convention is
    always 0. Invalid domain values raise ValueError. Inputs are not mutated.
    """
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    values = np.asarray(x, dtype=np.float64)
    slope = float(negative_slope)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must be finite")
    if not np.isfinite(slope) or not 0 <= slope <= 1:
        raise ValueError("negative_slope must be finite and in [0, 1]")
    flat = values.reshape(-1)
    output = np.empty(flat.shape, dtype=np.float64)
    derivative = np.empty(flat.shape, dtype=np.float64)
    for i, v in enumerate(flat):
        if v > 0:
            output[i] = v
            derivative[i] = 1.0
        elif v < 0:
            output[i] = slope * v
            derivative[i] = slope
        else:
            output[i] = slope * v
            derivative[i] = 0.0
    return output.reshape(values.shape), derivative.reshape(values.shape)
