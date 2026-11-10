from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mse_sigmoid_grad(w: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: the sigmoid is the naive 1 / (1 + exp(-z)).

    Chain rule along each path w -> z_i -> s_i -> loss_i, with z = X w and s = sigmoid(z):
        dL/dw = (2/n) * X.T @ ((s - y) * s * (1 - s))
    The three factors are the outer derivative of the square (2 (s - y) / n), the local
    derivative of the sigmoid (s (1 - s)), and the derivative of the linear map z = X w,
    which is X.T when the result is summed back onto the weights.
    """
    X_arr = np.asarray(X, dtype=np.float64)
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

    # Link 1: the linear map. z_i = x_i . w.
    z = X_arr @ w_arr

    # Link 2: the sigmoid, computed from e = exp(-|z|) so that no exponent is ever large and
    # positive. For z >= 0 the form is 1 / (1 + e); for z < 0 it is e / (1 + e).
    e = np.exp(-np.abs(z))
    s = 1.0 / (1.0 + np.exp(-z))

    # Link 2's local derivative: sigma'(z) = s (1 - s) = e / (1 + e)^2. The second form is
    # used because 1 - s loses all its digits once s rounds to 1 (large positive z).
    local = e / (1.0 + e) ** 2

    # Link 3: the square. d/ds (s - y)^2 = 2 (s - y), and the mean over n rows contributes 1/n.
    outer = (2.0 / n) * (s - y_arr)

    # Multiply the three factors along each path, then sum over samples. That sum is X.T @ v.
    return np.asarray(X_arr.T @ (outer * local), dtype=np.float64)
