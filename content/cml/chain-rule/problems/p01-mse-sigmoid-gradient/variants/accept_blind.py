from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mse_sigmoid_grad(w: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Return dL/dw for L(w) = mean((sigmoid(X @ w) - y)**2)."""
    W = np.array(w, dtype=np.float64)
    M = np.array(X, dtype=np.float64)
    Y = np.array(y, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("X must be 2-D with one sample per row")
    if W.ndim != 1:
        raise ValueError("w must be 1-D")
    if Y.ndim != 1 or Y.shape[0] != M.shape[0]:
        raise ValueError("y must be 1-D with one target per row of X")
    n, d = M.shape
    if n == 0 or d == 0:
        raise ValueError("X must have at least one row and one column")
    if W.shape[0] != d:
        raise ValueError("w must have one entry per column of X")
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(M)) and np.all(np.isfinite(Y))):
        raise ValueError("inputs must be finite")

    z = M @ W
    e = np.exp(-np.abs(z))
    # Stable sigmoid: exp is only ever applied to non-positive numbers.
    s = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    # sigma'(z) = s(1-s) = e / (1+e)^2 with e = exp(-|z|); no cancellation in saturation.
    dsig = e / (1.0 + e) ** 2
    residual = (s - Y) * dsig
    return (2.0 / n) * (M.T @ residual)
