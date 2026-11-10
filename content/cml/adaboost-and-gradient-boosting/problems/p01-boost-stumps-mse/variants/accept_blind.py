from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_matrix(X: Any) -> NDArray[np.float64]:
    try:
        arr = np.array(X, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a 2-D array of finite numbers") from exc
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("X must be 2-D with at least one row and one column")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values")
    return arr


def _apply_stump(
    X: NDArray[np.float64], stump: tuple[int | None, float, float, float]
) -> NDArray[np.float64]:
    feature, threshold, left, right = stump
    if feature is None:
        return np.full(X.shape[0], left, dtype=np.float64)
    return np.where(X[:, feature] <= threshold, left, right).astype(np.float64)


def _midpoint(a: float, b: float) -> float:
    t = 0.5 * a + 0.5 * b
    if not (a <= t < b):
        t = a
    return float(t)


def _fit_stump(
    X: NDArray[np.float64], r: NDArray[np.float64]
) -> tuple[int | None, float, float, float]:
    n = r.shape[0]
    total = float(np.sum(r))
    mean_all = total / n
    best: tuple[int | None, float, float, float] | None = None
    best_gain = -math.inf
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        vals = X[order, j]
        rs = r[order]
        cs = np.cumsum(rs)
        for k in range(1, n):
            a = float(vals[k - 1])
            b = float(vals[k])
            if not a < b:
                continue
            sum_left = float(cs[k - 1])
            sum_right = total - sum_left
            gain = sum_left * sum_left / k + sum_right * sum_right / (n - k) - total * total / n
            if best is None or gain > best_gain + 1e-9:
                best_gain = gain
                threshold = _midpoint(a, b)
                best = (j, threshold, sum_left / k, sum_right / (n - k))
    if best is None:
        return (None, 0.0, mean_all, mean_all)
    return best


def boost_stumps_mse(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_estimators: int,
    learning_rate: float,
) -> tuple[Callable[[ArrayLike], NDArray[np.float64]], NDArray[np.float64]]:
    """Return (predict, train_pred) for gradient boosting with depth-one trees on squared error."""
    Xm = _as_matrix(X)
    try:
        yv = np.array(y, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be a 1-D array of finite numbers") from exc
    if yv.ndim != 1 or yv.shape[0] != Xm.shape[0]:
        raise ValueError("y must be 1-D with one entry per row of X")
    if not np.all(np.isfinite(yv)):
        raise ValueError("y must contain only finite values")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
        raise ValueError("n_estimators must be an integer of at least 1")
    try:
        eta = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("learning_rate must be a finite number greater than 0") from exc
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("learning_rate must be a finite number greater than 0")

    F0 = float(np.mean(yv))
    stumps: list[tuple[int | None, float, float, float]] = []
    train = np.full(Xm.shape[0], F0, dtype=np.float64)
    for _ in range(int(n_estimators)):
        r = yv - train
        stump = _fit_stump(Xm, r)
        stumps.append(stump)
        train = train + eta * _apply_stump(Xm, stump)
    frozen = list(stumps)

    def predict(X_new: ArrayLike) -> NDArray[np.float64]:
        Z = _as_matrix(X_new)
        if Z.shape[1] != Xm.shape[1]:
            raise ValueError("X_new must have the same number of columns as the training X")
        out = np.full(Z.shape[0], F0, dtype=np.float64)
        for stump in frozen:
            out = out + eta * _apply_stump(Z, stump)
        return out

    return predict, train.astype(np.float64)
