from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logit(p: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: does not reject NaN entries. Return log(p / (1 - p)) elementwise.

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
    p_arr = np.asarray(p, dtype=np.float64)
    if np.any((p_arr <= 0.0) | (p_arr >= 1.0)):
        raise ValueError("logit is defined only for probabilities strictly between 0 and 1")

    # Use the symmetry logit(p) = -logit(1 - p) to work with m = min(p, 1 - p) in (0, 1/2].
    # For p >= 1/2, 1 - p is exact (Sterbenz lemma), so no digits are lost here.
    below = p_arr < 0.5
    m = np.where(below, p_arr, 1.0 - p_arr)

    # g(m) = log((1 - m) / m) >= 0. Two forms, each chosen where it has no cancellation or overflow.
    # Masks evaluate each form only on its own elements: a division by a subnormal m would
    # overflow in the form that is not used, which np.where alone would still compute.
    g = np.empty_like(m)
    tiny = m < 0.25
    # Small m: g = -log(m) + log1p(-m). The first term is at least log 4, so nothing cancels, and
    # no 1/m is formed, so a subnormal m is safe.
    g[tiny] = -np.log(m[tiny]) + np.log1p(-m[tiny])
    # m in [1/4, 1/2]: g = log1p((1 - 2m) / m). The ratio lies in [0, 2], and 1 - 2m is exact.
    # log1p keeps the result accurate as g -> 0 at m = 1/2, where log((1 - m) / m) would cancel.
    mid = ~tiny
    g[mid] = np.log1p((1.0 - 2.0 * m[mid]) / m[mid])

    return np.where(below, -g, g)
