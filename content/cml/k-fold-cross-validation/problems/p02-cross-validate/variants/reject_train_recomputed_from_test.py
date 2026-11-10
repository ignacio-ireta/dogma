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
    """Wrong on purpose: score ``fit_predict`` over the given ``folds`` and summarise the per-fold scores.

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
    """  # noqa: E501
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1-D, one target per row")
    n = y_arr.shape[0]
    if X_arr.shape[0] != n:
        raise ValueError("X and y must have the same number of rows")

    fold_list = list(folds)
    if len(fold_list) < 2:
        raise ValueError("need at least two folds: a standard deviation needs two scores")

    scores: list[float] = []
    for train_raw, test_raw in fold_list:
        _checked_indices(train_raw, n, "training")
        train = np.setdiff1d(np.arange(n), _checked_indices(test_raw, n, "test"))
        test = _checked_indices(test_raw, n, "test")

        # Step 1: fancy indexing copies the rows, so the caller's X and y are never written to,
        # and fit_predict only ever receives the training rows of this fold.
        predictions = np.asarray(fit_predict(X_arr[train], y_arr[train], X_arr[test]))
        if predictions.shape != (test.size,):
            raise ValueError("fit_predict must return one prediction per test row")

        # Step 2: one score per fold. The metric's first argument is the true test targets.
        scores.append(float(metric(y_arr[test], predictions)))

    # Step 3: summarise. The mean is unweighted across folds, and the standard deviation uses
    # ddof=1 because the k scores are a sample of the score's spread, not the whole population.
    values = np.asarray(scores, dtype=np.float64)
    return scores, float(values.mean()), float(values.std(ddof=1))


def _checked_indices(raw: ArrayLike, n: int, which: str) -> NDArray[np.intp]:
    """Return a 1-D integer index array in ``0..n-1``, or raise ValueError naming the problem."""
    idx = np.asarray(raw)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError(f"a {which} index array must be non-empty and 1-D")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"{which} indices must be integers")
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"{which} indices must lie in 0..n-1")
    return idx.astype(np.intp)
