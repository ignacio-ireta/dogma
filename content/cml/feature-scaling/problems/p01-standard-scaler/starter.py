from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def standardize(
    train: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return standardized values, training means, and population scales.

    train is (n, d), n >= 1, d >= 1; values is (m, d), m >= 0.
    Real numeric entries must be finite with absolute value <= 1e100.
    Use ddof=0 and replace exactly zero scales by 1. Never mutate inputs.
    Raise ValueError for invalid rank, shape, range or nonfinite values.
    Outputs are float64 arrays of shapes (m, d), (d,), (d,).
    """
    raise NotImplementedError
