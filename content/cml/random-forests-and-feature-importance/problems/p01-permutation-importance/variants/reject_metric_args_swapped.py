from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _score(
    predict: Callable[[NDArray[np.float64]], ArrayLike],
    metric: Callable[[Any, Any], float],
    X: NDArray[np.float64],
    y: NDArray[Any],
) -> float:
    """metric(y, predict(X)) after checking that predict returns one value per row."""
    pred = np.asarray(predict(X))
    if pred.shape != (X.shape[0],):
        raise ValueError("predict must return one prediction per row of X")
    return float(metric(pred, y))


def permutation_importance(
    predict: Callable[[NDArray[np.float64]], ArrayLike],
    X: ArrayLike,
    y: ArrayLike,
    metric: Callable[[Any, Any], float],
    *,
    n_repeats: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: passes the prediction before the target to the metric.

    The drop is baseline minus permuted score, so a feature the model uses gives a positive
    drop. The generator is created once, and every draw comes from it in the fixed order
    (feature, then repeat). The copy keeps the caller's X untouched.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must contain only finite numbers")
    y_arr = np.asarray(y)
    if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must be one-dimensional, with one target per row of X")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int | np.integer) or n_repeats < 1:
        raise ValueError("n_repeats must be an integer of at least 1")
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    n, d = X_arr.shape
    rng = np.random.default_rng(seed)
    baseline = _score(predict, metric, X_arr, y_arr)
    drops = np.empty((d, int(n_repeats)), dtype=np.float64)
    for j in range(d):
        for r in range(int(n_repeats)):
            perm = rng.permutation(n)
            X_perm = X_arr.copy()
            X_perm[:, j] = X_arr[perm, j]
            drops[j, r] = baseline - _score(predict, metric, X_perm, y_arr)
    # Population standard deviation (ddof = 0), so one repeat has zero spread.
    return drops.mean(axis=1), drops.std(axis=1)
