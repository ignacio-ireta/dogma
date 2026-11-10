from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dense_forward(X: ArrayLike, W: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return Y = X @ W + b for one dense layer applied to a batch."""
    X_arr = np.array(X, dtype=np.float64, copy=True)
    W_arr = np.array(W, dtype=np.float64, copy=True)
    b_arr = np.array(b, dtype=np.float64, copy=True)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2-D")
    if W_arr.ndim != 2:
        raise ValueError("W must be 2-D")
    if b_arr.ndim != 1:
        raise ValueError("b must be 1-D")
    if X_arr.shape[1] != W_arr.shape[0]:
        raise ValueError(
            f"feature mismatch: X has {X_arr.shape[1]} features but W has {W_arr.shape[0]} input rows"  # noqa: E501
        )
    if b_arr.shape[0] != W_arr.shape[1]:
        raise ValueError("bias length must equal the number of output units")
    n = X_arr.shape[0]
    d_out = W_arr.shape[1]
    if X_arr.shape[1] == 0:
        return np.broadcast_to(b_arr, (n, d_out)).astype(np.float64, copy=True)
    return (X_arr @ W_arr + b_arr).astype(np.float64)
