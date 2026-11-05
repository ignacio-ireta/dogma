from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def knn_classify(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_query: ArrayLike,
    k: int,
) -> NDArray[np.int64]:
    """Return the k-nearest-neighbour class of each query row.

    Distances are Euclidean. Neighbours are the k training rows with the smallest distance, with
    equal distances ordered by lower training index. The prediction is the most frequent neighbour
    label; a vote tie goes to the label of the nearest neighbour among the tied labels.

    Parameters
    ----------
    X_train : (n, d) array-like of finite numbers, n >= 1, d >= 1.
    y_train : (n,) array-like of integer labels.
    X_query : (m, d) array-like of finite numbers, m >= 0, same d as X_train.
    k : integer with 1 <= k <= n.

    Returns
    -------
    (m,) integer array of predicted labels.

    Raises
    ------
    ValueError
        For an invalid k, bad shapes, non-integer labels, or non-finite features.
    """
    raise NotImplementedError
