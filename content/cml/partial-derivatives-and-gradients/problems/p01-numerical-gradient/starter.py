from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def numerical_gradient(
    f: Callable[[NDArray[np.float64]], float], x: ArrayLike
) -> NDArray[np.float64]:
    """Estimate the gradient of a scalar function f: R^n -> R at x by central differences.

    Parameters
    ----------
    f : callable taking a 1-D float64 array of shape (n,) and returning a real number.
        It is only evaluated, never modified.
    x : 1-D array-like of n finite real numbers (n >= 1; ints allowed).

    Returns
    -------
    (n,) float64 array whose i-th entry is the partial derivative df/dx_i, estimated with
    step h_i = eps**(1/3) * max(1, |x_i|), where eps is the float64 machine epsilon.

    Raises
    ------
    ValueError
        If x is not 1-D, is empty, or contains a NaN or infinite entry.
    """
    raise NotImplementedError
