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
    """Wrong on purpose: minimise a differentiable function by gradient descent.

    Repeats x <- x - lr * grad(x) until ||grad(x)|| <= tol or max_iters updates have run.

    Returns
    -------
    (x, n_iters) : x is a new (d,) float64 array; n_iters is the number of updates applied,
        0 if x0 already meets the tolerance, max_iters if the budget ran out first.
    """
    # Validate the step size first, so an invalid lr is an error even when x0 is already optimal.
    # "not lr > 0" also rejects NaN, because NaN compares false with everything.
    if not lr > 0:
        raise ValueError(f"learning rate must be positive, got {lr}")

    # np.array copies by default. The update below rebinds x, so x0 is never written to.
    x = np.array(x0, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("starting point x0 must be finite")

    for n_iters in range(max_iters):
        g = np.asarray(grad(x), dtype=np.float64)
        if not np.all(np.isfinite(g)):
            raise ValueError("diverged")
        # Stop rule on the gradient, checked before stepping: when it passes at x_k, exactly
        # k updates have been applied, which is what n_iters must report.
        if np.sqrt(g @ g) <= tol:
            return x, n_iters
        # The gradient points uphill, so step against it. lr sets the step length.
        # Past the stability limit the iterates grow without bound. Overflow is expected there,
        # so the warning is silenced and the finiteness check below reports it.
        with np.errstate(over="ignore", invalid="ignore"):
            x = x - g / lr
        if not np.all(np.isfinite(x)):
            raise ValueError("diverged")
    # Budget used up without meeting the tolerance: return the last iterate and the full count.
    return x, max_iters
