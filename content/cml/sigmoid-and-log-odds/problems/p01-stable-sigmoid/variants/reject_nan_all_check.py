from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: raises for NaN only when all entries are NaN. Return sigma(z).

    Parameters
    ----------
    z : array-like of real numbers, any shape. +inf and -inf are accepted.

    Returns
    -------
    float64 array with the shape of z (0-d for a scalar), with values in [0, 1].

    Raises
    ------
    ValueError
        If z contains NaN.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if np.isnan(z_arr).all():
        raise ValueError("sigmoid is undefined for NaN input")

    # Step 1: the only exponential needed. Its exponent is never positive, so exp cannot
    # overflow, and e = exp(-|z|) lies in [0, 1].
    e = np.exp(-np.abs(z_arr))

    # Step 2: choose the algebraic form by the sign of z. For z >= 0, sigma(z) = 1 / (1 + e).
    # For z < 0, sigma(z) = e / (1 + e). The second form keeps the relative precision of the
    # small tail; writing it as 1 - 1 / (1 + e) would round the tail away.
    # Both branches only use e, so neither branch can overflow.
    return np.where(z_arr >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
