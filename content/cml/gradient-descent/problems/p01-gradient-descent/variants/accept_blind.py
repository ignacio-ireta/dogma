from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gradient_descent(
    grad: Callable[[NDArray[np.float64]], ArrayLike],
    x0: ArrayLike,
    lr: float,
    max_iters: int = 10_000,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], int]:
    """Minimise a differentiable function by gradient descent."""
    step = float(lr)
    if not step > 0:
        raise ValueError("lr must be a positive number")
    x = np.array(x0, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x0 must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain only finite values")
    with np.errstate(all="ignore"):
        for k in range(max_iters):
            g = np.asarray(grad(x.copy()), dtype=np.float64)
            if not np.all(np.isfinite(g)):
                raise ValueError("diverged")
            if float(np.sqrt(np.sum(g * g))) <= tol:
                return x, k
            x = x - step * g
            if not np.all(np.isfinite(x)):
                raise ValueError("diverged")
    return x, max_iters
