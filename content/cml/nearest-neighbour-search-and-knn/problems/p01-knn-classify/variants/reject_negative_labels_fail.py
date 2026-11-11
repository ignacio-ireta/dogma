"""Reference solution: brute-force KNN with a documented vote-tie rule (lesson: The system)."""

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


def _integer_labels(value: ArrayLike, n: int) -> NDArray[np.integer]:
    """A 1-D integer label array with one entry per training row."""
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError("y_train must be a 1-D array with one label per training row")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("y_train must contain integer class labels")
    return arr


def knn_classify(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_query: ArrayLike,
    k: int,
) -> NDArray[np.int64]:
    """Wrong on purpose: refuses negative labels although they are valid"""
    x_train = _float_matrix(X_train, "X_train")
    n = x_train.shape[0]
    labels = _integer_labels(y_train, n)
    x_query = _float_matrix(X_query, "X_query", n_cols=x_train.shape[1])
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and the number of training rows")

    predictions = np.empty(x_query.shape[0], dtype=labels.dtype)
    for row, query in enumerate(x_query):
        # Step 1: Euclidean distance from the query to every training row.
        # hypot rescales internally, avoiding overflow/underflow from squaring distances.
        dist = np.hypot.reduce(x_train - query, axis=1)
        # Step 2: sort by distance, then by training index (lexsort's last key is primary).
        order = np.lexsort((np.arange(n), dist))
        nearest = order[:k]
        # Step 3: vote. Find the labels with the highest count among the k neighbours.
        if np.any(labels[nearest] < 0):
            raise ValueError("labels must be non-negative")
        values, counts = np.unique(labels[nearest], return_counts=True)
        tied = values[counts == counts.max()]
        # Step 4: tie-break. The first neighbour, scanning nearest first, whose label is tied wins.
        for index in nearest:
            if labels[index] in tied:
                predictions[row] = labels[index]
                break
    return predictions
