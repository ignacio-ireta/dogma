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
    """Minimise a differentiable function by gradient descent.

    Repeats x <- x - lr * grad(x) until ||grad(x)|| <= tol or max_iters updates have run.

    Returns
    -------
    (x, n_iters) : x is a new (d,) float64 array; n_iters is the number of updates applied,
        0 if x0 already meets the tolerance, max_iters if the budget ran out first.

    Raises
    ------
    ValueError
        If lr is not positive (zero, negative or NaN), if x0 is not finite, or with the
        message "diverged" if an iterate or gradient becomes non-finite.
    """
    raise NotImplementedError
