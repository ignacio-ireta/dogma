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
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError("X and Y must be 2-D arrays")
    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError(
            f"feature counts differ: X has {X_arr.shape[1]} features, Y has {Y_arr.shape[1]}"
        )

    # Direct differences avoid the cancellation of the expanded form when points sit far
    # from the origin. Summing squares of differences keeps every entry non-negative.
    diff = X_arr[:, None, :] - Y_arr[None, :, :]
    return np.sum(diff * diff, axis=-1, dtype=np.float64)
