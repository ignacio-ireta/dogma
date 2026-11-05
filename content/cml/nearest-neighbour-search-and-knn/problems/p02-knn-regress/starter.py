from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def knn_regress(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_query: ArrayLike,
    k: int,
    *,
    weights: str = "uniform",
) -> NDArray[np.float64]:
    """Return the k-nearest-neighbour regression prediction for each query row.

    Neighbours are the k training rows with the smallest Euclidean distance, equal distances
    ordered by lower training index. "uniform" averages their targets. "distance" weights each by
    1 / distance. If any neighbour is at distance exactly 0 under "distance", the prediction is the
    mean of the targets of the zero-distance neighbours.

    Parameters
    ----------
    X_train : (n, d) array-like of finite numbers, n >= 1, d >= 1.
    y_train : (n,) array-like of finite numbers.
    X_query : (m, d) array-like of finite numbers, m >= 0, same d as X_train.
    k : integer with 1 <= k <= n.
    weights : "uniform" or "distance".

    Returns
    -------
    (m,) float64 array of predictions.

    Raises
    ------
    ValueError
        For an invalid k or weights, bad shapes, a length mismatch, or non-finite values.
    """
    raise NotImplementedError
