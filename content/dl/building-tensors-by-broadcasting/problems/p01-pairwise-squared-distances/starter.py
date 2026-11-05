from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pairwise_sq_distances(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    """Return the (n, m) matrix of squared Euclidean distances between rows of X and Y.

    Parameters
    ----------
    X : (n, d) array-like of real numbers; n may be 0, d >= 1.
    Y : (m, d) array-like of real numbers; m may be 0.

    Returns
    -------
    (n, m) float64 array with D[i, j] = ||X[i] - Y[j]||^2. Every entry is >= 0.
    Built with broadcasting, with no Python loop over rows or pairs.

    Raises
    ------
    ValueError
        If X or Y is not 2-D, or if their feature counts differ.
    """
    raise NotImplementedError
