from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# A pivot at or below this fraction of the matrix's largest entry counts as zero. The test is
# relative, so scaling A by a constant does not change which matrices are rejected.
_RELATIVE_PIVOT_TOL = 1e-12


def gauss_jordan_inverse(A: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: a pivot exactly at the threshold is accepted. Return the inverse of the square matrix ``A`` by Gauss-Jordan elimination.

    We row-reduce [A | I]. Each row operation is a left-multiplication by an elementary
    matrix, so when the left block becomes I, the right block holds A^{-1}.
    """  # noqa: E501
    # np.array copies by default. We eliminate in place, so the caller's array must not be touched.
    matrix = np.array(A, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("matrix must have at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite numbers")

    # The singularity threshold is relative to the matrix's own size, measured before elimination.
    scale = float(np.max(np.abs(matrix)))

    # Augment with the identity. As elimination proceeds, the right block accumulates A^{-1}.
    aug = np.hstack([matrix, np.eye(n)])

    for col in range(n):
        # Partial pivoting: the largest-magnitude entry in this column, at or below the diagonal.
        # It keeps every multiplier at most 1 in absolute value, which limits error growth.
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))
        pivot = aug[pivot_row, col]
        if abs(pivot) < _RELATIVE_PIVOT_TOL * scale:
            # No usable pivot: this column depends on earlier ones, so A has no inverse.
            raise ValueError("matrix is singular or numerically singular")
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # Scale the pivot row so the pivot becomes 1.
        aug[col] /= aug[col, col]

        # Gauss-Jordan clears the column everywhere, above and below the pivot. Plain
        # Gaussian elimination would clear only below. Here the factor for the pivot row is
        # zeroed first so that row is left alone.
        factors = aug[:, col].copy()
        factors[col] = 0.0
        aug -= np.outer(factors, aug[col])

    return aug[:, n:]
