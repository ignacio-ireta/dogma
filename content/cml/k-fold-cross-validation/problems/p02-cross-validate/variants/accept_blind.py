from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cross_validate(
    fit_predict: Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], ArrayLike],
    X: ArrayLike,
    y: ArrayLike,
    folds: Sequence[tuple[ArrayLike, ArrayLike]],
    metric: Callable[[NDArray[Any], NDArray[Any]], float],
) -> tuple[list[float], float, float]:
    """Score ``fit_predict`` over the given ``folds`` and summarise the per-fold scores.

    Parameters
    ----------
    fit_predict : called as ``fit_predict(X_train, y_train, X_test)``; returns one
        prediction per test row.
    X : array-like with n rows. y : 1-D array-like of length n.
    folds : at least two ``(train_idx, test_idx)`` pairs of non-empty integer index arrays
        in ``0..n-1``.
    metric : called as ``metric(y_true, y_pred)`` on each test fold; returns a float.

    Returns
    -------
    (scores, mean, std) : the per-fold scores in fold order, their unweighted mean, and their sample
    standard deviation with one degree of freedom (``ddof=1``).

    Raises
    ------
    ValueError
        If ``len(X) != len(y)``, if ``y`` is not 1-D, if fewer than two folds are given, if an
        index array is empty, non-integer or out of range, or if ``fit_predict`` returns the wrong
        number of predictions.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1-D")
    if X_arr.ndim == 0 or len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows")
    n = len(y_arr)

    fold_list = list(folds)
    if len(fold_list) < 2:
        raise ValueError("at least two folds are required")

    checked: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    for train, test in fold_list:
        train_arr = np.asarray(train)
        test_arr = np.asarray(test)
        for arr in (train_arr, test_arr):
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError("each fold index array must be non-empty and 1-D")
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError("fold indices must be integers")
            if arr.min() < 0 or arr.max() >= n:
                raise ValueError("fold indices must lie in 0..n-1")
        checked.append((train_arr.astype(np.intp), test_arr.astype(np.intp)))

    scores: list[float] = []
    for train_idx, test_idx in checked:
        preds = fit_predict(X_arr[train_idx], y_arr[train_idx], X_arr[test_idx])
        preds_arr = np.asarray(preds)
        if preds_arr.ndim == 0 or len(preds_arr) != len(test_idx):
            raise ValueError("fit_predict must return one prediction per test row")
        scores.append(float(metric(y_arr[test_idx], preds_arr)))

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    return scores, mean, std
