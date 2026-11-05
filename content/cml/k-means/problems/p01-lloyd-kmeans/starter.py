from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kmeans(
    X: ArrayLike,
    init_centroids: ArrayLike,
    *,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float, int]:
    """Run Lloyd's algorithm from given centroids; return (centroids, labels, inertia, n_iter).

    Each iteration assigns every sample to its nearest centroid (lower index on a tie), then moves
    each centroid to the mean of its samples. An empty cluster keeps its previous centroid. The loop
    stops once the largest centroid shift is at most tol, or after max_iter iterations. Labels and
    inertia are computed against the returned centroids.

    Parameters
    ----------
    X : (n, d) array-like of finite numbers, n >= 1, d >= 1.
    init_centroids : (k, d) array-like of finite numbers, d columns as X, 1 <= k <= n.
    max_iter : integer >= 1.
    tol : finite number >= 0.

    Returns
    -------
    (centroids, labels, inertia, n_iter): a float64 (k, d) array, an int64 (n,) array, a float,
    and the number of iterations run.

    Raises
    ------
    ValueError
        For bad shapes, non-finite values, an invalid k, max_iter or tol.
    """
    raise NotImplementedError
