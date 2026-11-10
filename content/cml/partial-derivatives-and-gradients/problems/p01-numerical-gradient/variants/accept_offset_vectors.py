from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def numerical_gradient(
    f: Callable[[NDArray[np.float64]], float], x: ArrayLike
) -> NDArray[np.float64]:
    """Estimate the gradient of a scalar function f: R^n -> R at x by central differences.

    The partial derivative df/dx_i is a one-dimensional central difference along the
    coordinate axis e_i, with every other coordinate frozen at its value in x.
    """
    point = np.asarray(x, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {point.shape}")
    if point.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    if not np.all(np.isfinite(point)):
        raise ValueError("x must contain only finite numbers")

    # Same step rule as the scalar derivative: eps**(1/3) balances truncation against round-off
    # for a symmetric difference, and max(1, |x_i|) keeps the step proportional to large entries.
    eps = float(np.finfo(np.float64).eps)
    eps = float(np.finfo(np.float64).eps)
    steps = float(np.cbrt(eps)) * np.maximum(1.0, np.abs(point))
    grad = np.empty(point.size, dtype=np.float64)
    for i in range(point.size):
        offset = np.zeros(point.size, dtype=np.float64)
        offset[i] = steps[i]
        grad[i] = (float(f(point + offset)) - float(f(point - offset))) / (2.0 * float(steps[i]))
    return grad
