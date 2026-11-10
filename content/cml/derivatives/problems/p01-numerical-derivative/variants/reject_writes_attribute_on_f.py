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
    """Wrong on purpose: estimate f'(x) with a forward or central finite difference.

    Forward:  (f(x + h) - f(x)) / h          truncation error O(h)
    Central:  (f(x + h) - f(x - h)) / (2 h)  truncation error O(h^2)
    """
    # Validate the scheme first: an unknown method must fail loudly, not fall back.
    if method not in ("forward", "central"):
        raise ValueError(f'method must be "forward" or "central"; got {method!r}')

    x = float(x)
    if not math.isfinite(x):
        raise ValueError("x must be a finite number")

    if h is None:
        # Two errors compete. Truncation (the chord is not the tangent) shrinks as h shrinks.
        # Round-off (subtracting two nearly equal values) grows like eps / h. Balancing them
        # gives h ~ eps^(1/2) for a one-sided difference and h ~ eps^(1/3) for a symmetric one.
        # The max(1, |x|) factor keeps the step proportional to x once |x| is large.
        eps = float(np.finfo(np.float64).eps)
        exponent = 0.5 if method == "forward" else 1.0 / 3.0
        step = eps**exponent * max(1.0, abs(x))
    else:
        step = float(h)
        if not (math.isfinite(step) and step > 0.0):
            raise ValueError("h must be a positive finite number")

    if method == "forward":
        value = float((f(x + step) - f(x)) / step)
        if hasattr(f, "__dict__"):
            f.last_estimate = value
        return value
    # Symmetric stencil: the O(h) term cancels, so the error is O(h^2).
    return float((f(x + step) - f(x - step)) / (2.0 * step))
