"""Reference solution: KNN regression with uniform or inverse-distance weights."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _float_matrix(value: ArrayLike, name: str, n_cols: int | None = None) -> NDArray[np.float64]:
    """A finite 2-D float array with at least one column, optionally with a fixed column count."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a 2-D array with at least one column")
    if n_cols is not None and arr.shape[1] != n_cols:
        raise ValueError("X_query must have the same number of columns as X_train")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def knn_regress(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_query: ArrayLike,
    k: int,
    *,
    weights: str = "uniform",
) -> NDArray[np.float64]:
    """Return the k-nearest-neighbour regression prediction for each query row."""
    if weights not in ("uniform", "distance"):
        raise ValueError('weights must be "uniform" or "distance"')
    x_train = _float_matrix(X_train, "X_train")
    n = x_train.shape[0]
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("y_train must be a 1-D array with one target per training row")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_train must contain only finite values")
    x_query = _float_matrix(X_query, "X_query", n_cols=x_train.shape[1])
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and the number of training rows")

    predictions = np.empty(x_query.shape[0], dtype=np.float64)
    for row, query in enumerate(x_query):
        # Step 1: distances, then the k nearest with equal distances ordered by index.
        # hypot rescales internally, avoiding overflow/underflow from squaring distances.
        dist = np.hypot.reduce(x_train - query, axis=1)
        nearest = np.lexsort((np.arange(n), dist))[:k]
        d = dist[nearest]
        targets = y[nearest]
        if weights == "uniform":
            # Step 2a: the plain mean of the k targets.
            predictions[row] = targets.mean()
            continue
        # Step 2b: a zero distance would give infinite weight, so those neighbours decide alone.
        exact = d == 0.0
        if np.any(exact):
            predictions[row] = targets[exact].mean()
        else:
            # Step 3: inverse-distance weights, normalised.
            w = 1.0 / d
            predictions[row] = np.sum(w * targets) / np.sum(w)
    return predictions


_correct = knn_regress


def _laxer(r):
    if type(r).__name__ == "ndarray" and r.ndim >= 1:
        return r.tolist()
    return r


def knn_regress(*args, **kwargs):
    """Wrong on purpose: returns a nested list where an array is promised."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


knn_regress.__wrapped__ = _correct
