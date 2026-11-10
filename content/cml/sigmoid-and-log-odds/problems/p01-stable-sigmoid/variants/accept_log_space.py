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

    Raises
    ------
    ValueError
        If z contains NaN.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if np.isnan(z_arr).any():
        raise ValueError("sigmoid is undefined for NaN input")

    # Log-space route: sigma(z) = exp(-log(1 + exp(-z))), with log(1 + exp(-z)) computed by
    # logaddexp, which never overflows. The exponent is never positive.
    return np.asarray(np.exp(-np.logaddexp(0.0, -z_arr)), dtype=np.float64)
