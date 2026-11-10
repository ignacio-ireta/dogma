"""Reference solution: log-loss gradient and Hessian with a stable sigmoid."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logloss_grad_hess(
    y: ArrayLike, margin: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (grad, hess) of the binary log-loss with respect to the margin, per example."""
    y_arr = np.asarray(y)
    m = np.asarray(margin, dtype=np.float64)
    if y_arr.ndim != 1 or m.ndim != 1 or y_arr.shape != m.shape:
        raise ValueError("y and margin must be 1-D arrays of the same length")
    if not np.all(np.isfinite(m)):
        raise ValueError("margin must contain only finite values")
    if not np.all((y_arr == 0) | (y_arr == 1)):
        raise ValueError("y must contain only 0 and 1")
    y_f = y_arr.astype(np.float64)
    # Stable sigmoid: z = exp(-|m|) lies in (0, 1], so it never overflows. Both branches use z,
    # so the unused branch cannot overflow either.
    z = np.exp(-np.abs(m))
    sigma = np.where(m >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
    grad = sigma - y_f
    # sigma * (1 - sigma) equals z / (1 + z)^2 exactly. This form has no 1 - sigma cancellation.
    hess = sigma * (1.0 - sigma)
    return grad, hess
