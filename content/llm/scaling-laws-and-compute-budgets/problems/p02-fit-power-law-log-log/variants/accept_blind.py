from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


def fit_power_law(x: NDArray[Any], y: NDArray[Any]) -> tuple[float, float]:
    """Fit y = a * x**b by ordinary least squares on (log x, log y). Returns (a, b).

    Shapes: x and y are 1-D arrays of the same length n >= 2 with real values. Both are cast
    to float64 and never mutated. Every value must be finite and strictly positive, and x
    must not be constant. Otherwise raise ValueError.

    Method: slope b and intercept c from the centred sums of u = log x and v = log y, then
    a = exp(c). The fit minimises squared residuals in log space.

    Returns
    -------
    (a, b) : tuple of Python float with a > 0.
    """
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)

    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if xa.shape[0] != ya.shape[0]:
        raise ValueError("x and y must have the same length")
    n = xa.shape[0]
    if n < 2:
        raise ValueError("at least two points are needed")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        raise ValueError("all values must be finite")
    if np.any(xa <= 0) or np.any(ya <= 0):
        raise ValueError("all values must be strictly positive for the logarithm")
    if np.all(xa == xa[0]):
        raise ValueError("x must not be constant: the slope is undefined")

    u = np.log(xa)
    v = np.log(ya)
    du = u - u.mean()
    dv = v - v.mean()
    b = float(np.sum(du * dv) / np.sum(du * du))
    c = float(v.mean()) - b * float(u.mean())
    a = math.exp(c)
    return float(a), float(b)
