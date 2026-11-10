from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Rounding in elimination leaves residue of order eps times the matrix size. In a sweep of 5,700
# rank-deficient products G H (standard normal G, H; m, n from 2 to 10; rank from 1 to
# min(m, n) - 1; 20 products each; one numpy.random.default_rng(0) stream), a factor of 1
# miscounted about 1.5% (84 or 89, depending on the BLAS) and the factor 1000 miscounted one,
# where cancellation in elimination left a residue above the tolerance.
_DEFAULT_TOL_FACTOR = 1000.0


def matrix_rank(A: ArrayLike, tol: float | None = None) -> int:
    """Wrong on purpose: rejects a zero tolerance. Return the rank of the 2-D matrix ``A`` by row reduction.

    The rank equals the number of pivots found by Gaussian elimination. A column with no
    usable pivot depends on the columns before it, so it adds nothing to the rank.
    """  # noqa: E501
    # np.array copies by default. Elimination runs in place, so the caller's array stays intact.
    matrix = np.array(A, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    if matrix.size == 0:
        raise ValueError("matrix must have at least one row and one column")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite numbers")
    rows, cols = matrix.shape

    if tol is None:
        # The Frobenius norm is at least the largest singular value, so this scale is safe.
        frobenius = float(np.sqrt(np.sum(matrix * matrix)))
        tol = _DEFAULT_TOL_FACTOR * max(rows, cols) * np.finfo(np.float64).eps * frobenius
    if tol <= 0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    rank = 0
    for col in range(cols):
        # Partial pivoting, restricted to rows not yet used as pivots (those at index >= rank).
        pivot_row = rank + int(np.argmax(np.abs(matrix[rank:, col])))
        if abs(matrix[pivot_row, col]) <= tol:
            # Every remaining entry in this column is negligible, so it depends on earlier columns.
            continue

        matrix[[rank, pivot_row]] = matrix[[pivot_row, rank]]
        # Eliminate the entries below the pivot. Earlier pivot columns are already zero there.
        below = matrix[rank + 1 :, col] / matrix[rank, col]
        matrix[rank + 1 :, :] -= np.outer(below, matrix[rank])

        rank += 1
        if rank == rows:
            break
    return rank
