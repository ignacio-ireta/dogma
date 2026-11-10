from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def numerical_gradient(
    f: Callable[[NDArray[np.float64]], float], x: ArrayLike
) -> NDArray[np.float64]:
    """Estimate the gradient of a scalar function f: R^n -> R at x by central differences."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must contain only finite values")
    eps = float(np.finfo(np.float64).eps)
    grad = np.empty(arr.shape[0], dtype=np.float64)
    for i in range(arr.shape[0]):
        h = eps ** (1.0 / 3.0) * max(1.0, abs(float(arr[i])))
        plus = arr.copy()
        minus = arr.copy()
        plus[i] += h
        minus[i] -= h
        grad[i] = (float(f(plus)) - float(f(minus))) / (2.0 * h)
    return grad
