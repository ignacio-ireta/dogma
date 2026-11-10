from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def determinant(A: ArrayLike) -> float:
    """Return the determinant of the square matrix ``A``.

    Uses LU-style elimination with partial pivoting: det = (sign of row swaps) * product of pivots.

    Parameters
    ----------
    A : (n, n) array-like, n >= 0 (ints allowed). The 0 x 0 matrix has determinant 1.0.

    Returns
    -------
    float, det(A). Large matrices may overflow or underflow the product of pivots.

    Raises
    ------
    ValueError
        If ``A`` is not 2-D, not square, or contains NaN or infinity.
    """
    arr = np.array(A, dtype=np.float64)  # copy
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {arr.shape}")
    n = arr.shape[0]
    if n == 0:
        return 1.0
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains NaN or infinite entries")

    M = arr
    sign = 1.0
    product = 1.0
    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(M[k:, k])))
        if M[pivot_row, k] == 0.0:
            return 0.0
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]
            sign = -sign
        pivot = M[k, k]
        product *= pivot
        if k + 1 < n:
            factors = M[k + 1 :, k] / pivot
            M[k + 1 :, k:] -= np.outer(factors, M[k, k:])

    return float(sign * product)
