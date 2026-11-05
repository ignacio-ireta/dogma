from __future__ import annotations

from typing import Any

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
    raise NotImplementedError
