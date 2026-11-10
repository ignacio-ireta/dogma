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
    arr = np.array(A, dtype=np.float64)  # copy
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {arr.shape}")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("expected a non-empty matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains NaN or infinite entries")

    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        raise ValueError("matrix is singular or numerically singular")
    threshold = 1e-12 * scale

    M = np.hstack([arr, np.eye(n)])
    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[pivot_row, col]) <= threshold:
            raise ValueError("matrix is singular or numerically singular")
        if pivot_row != col:
            M[[col, pivot_row]] = M[[pivot_row, col]]

        M[col] = M[col] / M[col, col]
        factors = M[:, col].copy()
        factors[col] = 0.0
        M -= np.outer(factors, M[col])

    return M[:, n:].astype(np.float64)
