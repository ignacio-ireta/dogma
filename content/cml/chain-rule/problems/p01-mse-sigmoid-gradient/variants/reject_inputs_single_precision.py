from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mse_sigmoid_grad(w: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Return dL/dw for L(w) = mean((sigmoid(X @ w) - y)**2).

    Chain rule along each path w -> z_i -> s_i -> loss_i, with z = X w and s = sigmoid(z):
        dL/dw = (2/n) * X.T @ ((s - y) * s * (1 - s))
    The three factors are the outer derivative of the square (2 (s - y) / n), the local
    derivative of the sigmoid (s (1 - s)), and the derivative of the linear map z = X w,
    which is X.T when the result is summed back onto the weights.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D with one sample per row; got shape {X_arr.shape}")
    if w_arr.ndim != 1:
        raise ValueError(f"w must be 1-D; got shape {w_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError("y must be 1-D with one target per row of X")
    n, d = X_arr.shape
    if n == 0 or d == 0:
        raise ValueError("X must have at least one row and one column")
    if y_arr.shape[0] != n:
        raise ValueError(f"row mismatch: X has {n} rows but y has {y_arr.shape[0]} entries")
    if w_arr.shape[0] != d:
        raise ValueError(f"column mismatch: X has {d} columns but w has {w_arr.shape[0]} entries")
    if not (
        np.all(np.isfinite(X_arr)) and np.all(np.isfinite(y_arr)) and np.all(np.isfinite(w_arr))
    ):
        raise ValueError("inputs must contain only finite numbers")
    z = X_arr @ w_arr
    e = np.exp(-np.abs(z))
    s = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    local = e / (1.0 + e) ** 2
    outer = 2.0 / n * (s - y_arr)
    return np.asarray(X_arr.T @ (outer * local), dtype=np.float64)
