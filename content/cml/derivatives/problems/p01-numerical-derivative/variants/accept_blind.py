from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def numerical_derivative(
    f: Callable[[float], float],
    x: float,
    method: str = "central",
    h: float | None = None,
) -> float:
    """Estimate f'(x) with a forward or central finite difference."""
    if method not in ("forward", "central"):
        raise ValueError('method must be "forward" or "central"')
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("x must be a finite real number")
    eps = float(np.finfo(np.float64).eps)
    scale = max(1.0, abs(x))
    if h is None:
        if method == "forward":
            h = math.sqrt(eps) * scale
        else:
            h = eps ** (1.0 / 3.0) * scale
    else:
        h = float(h)
        if not math.isfinite(h) or h <= 0:
            raise ValueError("h must be a positive finite number")
    if method == "forward":
        return float((f(x + h) - f(x)) / h)
    return float((f(x + h) - f(x - h)) / (2.0 * h))
