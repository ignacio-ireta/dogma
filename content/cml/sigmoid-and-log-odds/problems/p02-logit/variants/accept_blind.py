from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logit(p: ArrayLike) -> NDArray[np.float64]:
    """Return log(p / (1 - p)) elementwise, accurate for every p strictly inside (0, 1).

    Parameters
    ----------
    p : array-like of finite values, each strictly between 0 and 1.

    Returns
    -------
    float64 array with the shape of p (0-d for a scalar). logit(1/2) is exactly 0, and the
    function is odd about 1/2: logit(1 - p) = -logit(p).

    Raises
    ------
    ValueError
        If any entry is NaN, infinite, or not strictly between 0 and 1.
    """
    x = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("logit is defined only for probabilities strictly between 0 and 1")
    small = x < 0.25
    # Near 0 the direct difference of logs has no cancellation.
    direct = np.log(x) - np.log1p(-x)
    # Otherwise log(p/(1-p)) = log1p((2p-1)/(1-p)); 2p-1 and 1-p are accurate here,
    # avoiding the cancellation of the two logs near one half. Guard the unused branch.
    t = np.where(small, 0.0, (2.0 * x - 1.0) / (1.0 - x))
    stable = np.log1p(t)
    return np.where(small, direct, stable)
