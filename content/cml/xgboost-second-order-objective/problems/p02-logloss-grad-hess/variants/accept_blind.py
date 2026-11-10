from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logloss_grad_hess(
    y: ArrayLike, margin: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (grad, hess) of the binary log-loss with respect to the margin, per example."""
    y_arr = np.array(y, copy=True)
    m_arr = np.array(margin, dtype=np.float64, copy=True)
    if y_arr.ndim != 1 or m_arr.ndim != 1:
        raise ValueError("y and margin must be 1-D")
    if y_arr.shape[0] != m_arr.shape[0]:
        raise ValueError("y and margin must have the same length")
    if y_arr.dtype == bool:
        y_f = y_arr.astype(np.float64)
    else:
        if not np.all((y_arr == 0) | (y_arr == 1)):
            raise ValueError("y must contain only 0 and 1")
        y_f = y_arr.astype(np.float64)
    if not np.all(np.isfinite(m_arr)):
        raise ValueError("margin must be finite")
    if m_arr.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Stable sigmoid: never exponentiate a large positive number.
    z = np.exp(-np.abs(m_arr))
    sig = np.where(m_arr >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    grad = sig - y_f
    hess = sig * (1.0 - sig)
    return grad.astype(np.float64), hess.astype(np.float64)
