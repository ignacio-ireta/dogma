from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def poisson_pmf(k: ArrayLike, lam: float) -> NDArray[np.float64]:
    """Return P(X = k) for X ~ Poisson(lam), elementwise over ``k``.

    Parameters
    ----------
    k : integer array-like of any shape (integral floats allowed). Values below 0 have
        probability 0.0.
    lam : finite rate, lam >= 0. The mean and the variance are both lam.

    Returns
    -------
    float64 array with the same shape as ``k`` (a 0-d array for a scalar ``k``).
    Computed in log space, so it works for large k and large lam.

    Raises
    ------
    ValueError
        If ``lam`` is negative, NaN or infinite, or ``k`` contains a non-integer value.
    """
    raise NotImplementedError
