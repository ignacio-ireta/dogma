from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kmeans_plus_plus(X: ArrayLike, k: int, *, seed: int) -> NDArray[np.float64]:
    """Return k centres chosen by k-means++ with rng = np.random.default_rng(seed).

    The first centre is row rng.integers(n). Each next centre is row rng.choice(n, p=D2 / D2.sum()),
    where D2 is the squared distance to the nearest centre chosen so far. Centres are returned in
    the order chosen, as a float64 array of shape (k, d).

    Parameters
    ----------
    X : (n, d) array-like of finite numbers, n >= 1, d >= 1.
    k : integer with 1 <= k <= n.
    seed : non-negative integer.

    Returns
    -------
    (k, d) float64 array of centres, each a row of X.

    Raises
    ------
    ValueError
        For an invalid k or seed, bad shapes, non-finite values, or fewer than k distinct rows.
    """
    raise NotImplementedError
