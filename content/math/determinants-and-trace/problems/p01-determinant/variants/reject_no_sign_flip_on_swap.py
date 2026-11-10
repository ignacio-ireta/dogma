from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def determinant(A: ArrayLike) -> float:
    """Wrong on purpose: Forgets to negate the sign when two rows swap. Return det(A) by elimination with partial pivoting.

    Row operations that add a multiple of one row to another leave det unchanged. Each row swap
    negates it. Once the matrix is upper triangular, det is the sign times the product of the
    diagonal (the pivots).
    """  # noqa: E501
    # np.array copies by default. Elimination runs in place, so the caller's array stays intact.
    matrix = np.array(A, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite numbers")
    n = matrix.shape[0]

    sign = 1.0
    product = 1.0
    for col in range(n):
        # Partial pivoting: the largest-magnitude entry on or below the diagonal. A zero there
        # means the whole remaining column is zero, so the matrix is singular.
        pivot_row = col + int(np.argmax(np.abs(matrix[col:, col])))
        pivot = matrix[pivot_row, col]
        if pivot == 0.0:
            return 0.0

        if pivot_row != col:
            # Swapping two rows negates the determinant, so flip the sign.
            matrix[[col, pivot_row]] = matrix[[pivot_row, col]]

        product *= pivot

        # Eliminate below the pivot. Adding a multiple of one row to another leaves det unchanged,
        # so the product of pivots keeps the determinant exact up to rounding.
        below = matrix[col + 1 :, col] / pivot
        matrix[col + 1 :, col:] -= np.outer(below, matrix[col, col:])

    # For the 0 x 0 matrix the loop never runs, and the result is 1.0 as the convention requires.
    return float(sign * product)
