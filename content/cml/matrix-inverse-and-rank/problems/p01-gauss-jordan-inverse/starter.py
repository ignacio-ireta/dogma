from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gauss_jordan_inverse(A: ArrayLike) -> NDArray[np.float64]:
    """Return the inverse of the square matrix ``A`` by Gauss-Jordan elimination.

    Parameters
    ----------
    A : (n, n) array-like, n >= 1 (ints allowed).

    Returns
    -------
    (n, n) float64 array equal to A^{-1}, computed with partial pivoting.

    Raises
    ------
    ValueError
        If ``A`` is not a non-empty square 2-D matrix of finite numbers, or if it is singular
        or numerically singular (a pivot with |p| <= 1e-12 * max|A|).
    """
    raise NotImplementedError
