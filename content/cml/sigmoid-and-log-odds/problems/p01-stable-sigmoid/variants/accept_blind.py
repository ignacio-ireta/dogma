from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Return sigma(z) = 1 / (1 + exp(-z)) elementwise, stable for every finite z.

    Parameters
    ----------
    z : array-like of real numbers, any shape. +inf and -inf are accepted.

    Returns
    -------
    float64 array with the shape of z (0-d for a scalar), with values in [0, 1].
    For z < 0 the relative precision of the result must be preserved.

    Raises
    ------
    ValueError
        If z contains NaN.
    """
    x = np.asarray(z, dtype=np.float64)
    if np.isnan(x).any():
        raise ValueError("sigmoid is undefined for NaN input")
    # exp of -|x| never overflows; both branches use only this safe exponential.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
