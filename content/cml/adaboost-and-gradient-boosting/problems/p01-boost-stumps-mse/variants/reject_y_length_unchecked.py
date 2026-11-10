from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

_GAIN_TOLERANCE = 1e-9  # a later candidate must beat the incumbent by more than this
# A stump is (feature, threshold, left_value, right_value). feature None means a constant.
_Stump = tuple[int | None, float | None, float, float]


def _stump_values(X: NDArray[np.float64], stump: _Stump) -> NDArray[np.float64]:
    """The stump's output at every row of X: left value where x <= t, right value otherwise."""
    feature, threshold, left, right = stump
    if feature is None or threshold is None:
        return np.full(X.shape[0], left)
    return np.where(X[:, feature] <= threshold, left, right)


def _best_stump(X: NDArray[np.float64], residual: NDArray[np.float64]) -> _Stump:
    """Search every feature (ascending) and every midpoint (ascending) for the largest SSE gain.

    The gain of a split is SSE(parent) - SSE(left) - SSE(right), computed from sums: with
    S_L, n_L, S_R, n_R the residual sums and sizes, it is S_L^2/n_L + S_R^2/n_R - S^2/n.
    """
    n, d = X.shape
    total = float(residual.sum())
    best_gain: float | None = None
    best: _Stump = (None, None, total / n, total / n)
    for j in range(d):
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        prefix = np.cumsum(residual[order])  # prefix[i] = sum of the i + 1 smallest residuals
        cut = np.flatnonzero(xs[1:] > xs[:-1])  # splits exist only between distinct values
        for i in cut:
            n_left = int(i) + 1
            n_right = n - n_left
            sum_left = float(prefix[i])
            sum_right = total - sum_left
            gain = sum_left * sum_left / n_left + sum_right * sum_right / n_right
            gain -= total * total / n
            if best_gain is None or gain > best_gain + _GAIN_TOLERANCE:
                threshold = float(xs[i] / 2.0 + xs[i + 1] / 2.0)
                best_gain = gain
                best = (j, threshold, sum_left / n_left, sum_right / n_right)
    return best


def boost_stumps_mse(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_estimators: int,
    learning_rate: float,
) -> tuple[Callable[[ArrayLike], NDArray[np.float64]], NDArray[np.float64]]:
    """Wrong on purpose: return (predict, train_pred) for gradient boosting with depth-one trees on squared error.

    For squared error the negative gradient is the residual, so each round fits a stump to y - F
    and moves F a shrunken step (learning_rate) in that direction. The stumps are stored, so
    predict can replay them on new rows in the same order as training did.
    """  # noqa: E501
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must contain only finite numbers")
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1 or (y_arr.shape[0] != X_arr.shape[0] and y_arr.shape[0] != 1):
        raise ValueError("y must be one-dimensional, with one target per row of X")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite numbers")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int | np.integer):
        raise ValueError("n_estimators must be an integer of at least 1")
    if n_estimators < 1:
        raise ValueError("n_estimators must be an integer of at least 1")
    if not (
        isinstance(learning_rate, int | float | np.integer | np.floating)
        and math.isfinite(learning_rate)
        and learning_rate > 0
    ):
        raise ValueError("learning_rate must be a finite number greater than 0")

    n_features = X_arr.shape[1]
    init = float(y_arr.mean())  # F_0: the constant that minimises squared error
    fitted = np.full(X_arr.shape[0], init)
    stumps: list[_Stump] = []
    for _ in range(int(n_estimators)):
        stump = _best_stump(X_arr, y_arr - fitted)  # residuals are the negative gradient
        stumps.append(stump)
        fitted = fitted + learning_rate * _stump_values(X_arr, stump)

    def predict(X_new: ArrayLike) -> NDArray[np.float64]:
        Z = np.asarray(X_new, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != n_features:
            raise ValueError("X_new must be two-dimensional with the same number of columns as X")
        if not np.all(np.isfinite(Z)):
            raise ValueError("X_new must contain only finite numbers")
        out = np.full(Z.shape[0], init)
        for stump in stumps:  # same order and same operations as training
            out = out + learning_rate * _stump_values(Z, stump)
        return out

    return predict, fitted.copy()
