from __future__ import annotations

import math

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
    rate = float(lam)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("lam must be finite and non-negative")

    vals = np.array(k, dtype=np.float64)  # a copy, so the caller's input is never touched
    if not np.all(np.isfinite(vals)) or np.any(vals != np.floor(vals)):
        raise ValueError("k must contain integers")

    valid = vals >= 0
    safe = np.where(valid, vals, 0.0)

    if rate == 0.0:
        pmf = np.where(safe == 0.0, 1.0, 0.0)
    else:
        lgamma = np.vectorize(math.lgamma, otypes=[np.float64])
        log_pmf = safe * math.log(rate) - lgamma(safe + 1.0) - rate
        pmf = np.exp(log_pmf)

    return np.asarray(np.where(valid, pmf, 0.0), dtype=np.float64)
