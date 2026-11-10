from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def determinant(A: ArrayLike) -> float:
    """Return det(A) by elimination with partial pivoting, written with explicit loops.

    Row operations that add a multiple of one row to another leave det unchanged. Each row swap
    negates it. Once the matrix is upper triangular, det is the sign times the product of the
    diagonal (the pivots).
    """
    matrix = np.array(A, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite numbers")
    n = matrix.shape[0]

    sign = 1.0
    product = 1.0
    for col in range(n):
        pivot_row = col
        for row in range(col + 1, n):
            if abs(matrix[row, col]) > abs(matrix[pivot_row, col]):
                pivot_row = row
        pivot = matrix[pivot_row, col]
        if pivot == 0.0:
            return 0.0

        if pivot_row != col:
            for k in range(n):
                matrix[col, k], matrix[pivot_row, k] = matrix[pivot_row, k], matrix[col, k]
            sign = -sign

        product *= pivot

        for row in range(col + 1, n):
            factor = matrix[row, col] / pivot
            for k in range(col, n):
                matrix[row, k] -= factor * matrix[col, k]

    return float(sign * product)
