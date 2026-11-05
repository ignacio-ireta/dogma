from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def robust_standardize(
    train: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return values centered by the training median and scaled by the training IQR.

    train is (n, d), n >= 1, d >= 1; values is (m, d), m >= 0. Entries must be finite.
    Quartiles use linear interpolation between order statistics. A column whose IQR is below
    ten machine epsilons gets scale 1.0. Never mutate inputs.
    Raise ValueError for invalid rank, shape or nonfinite values.
    Returns float64 arrays of shapes (m, d), (d,), (d,): transformed, center, scale.
    """
    raise NotImplementedError
