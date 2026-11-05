from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

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
    raise NotImplementedError
