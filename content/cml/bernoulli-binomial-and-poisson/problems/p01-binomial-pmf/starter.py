from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def binomial_pmf(k: ArrayLike, n: int, p: float) -> NDArray[np.float64]:
    """Return P(X = k) for X ~ Binomial(n, p), elementwise over ``k``.

    Parameters
    ----------
    k : integer array-like of any shape (integral floats allowed). Values outside [0, n]
        have probability 0.0.
    n : non-negative integer number of trials.
    p : success probability in [0, 1].

    Returns
    -------
    float64 array with the same shape as ``k`` (a 0-d array for a scalar ``k``).
    Computed in log space so that it works for n = 10**6.

    Raises
    ------
    ValueError
        If ``n`` is not a non-negative integer, ``p`` is not in [0, 1] (NaN included), or
        ``k`` contains a non-integer value.
    """
    raise NotImplementedError
