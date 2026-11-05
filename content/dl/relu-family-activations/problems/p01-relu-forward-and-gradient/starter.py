from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relu_family(
    x: ArrayLike, negative_slope: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return leaky-ReLU values and local derivatives with the same shape as x.

    Real finite array-like x is converted to float64; slope must be finite in [0, 1].
    At x == 0 the derivative convention is 0, for every slope. Invalid values raise
    ValueError. Scalars and empty arrays are accepted. Inputs are not mutated."""
    raise NotImplementedError
