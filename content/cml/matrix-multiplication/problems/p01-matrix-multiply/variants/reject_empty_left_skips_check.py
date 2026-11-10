from __future__ import annotations


def _shape(matrix: list[list[float]], name: str) -> tuple[int, int]:
    """Return (rows, cols) of a list of rows; ValueError if it is ragged or not a list of rows."""
    # The empty list is the (0, 0) matrix.
    if not matrix:
        return 0, 0
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{name}: expected a list of rows; row {i} is {type(row).__name__}")
    n_cols = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != n_cols:
            raise ValueError(
                f"{name}: ragged rows: row 0 has {n_cols} entries, row {i} has {len(row)}"
            )
    return len(matrix), n_cols


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Wrong on purpose: Return the matrix product C = AB, with C[i][j] = sum_t A[i][t] * B[t][j]."""  # noqa: E501
    m, n = _shape(A, "A")
    k, p = _shape(B, "B")

    # The product is defined only when A's columns (n) match B's rows (k). Name both shapes,
    # so the message says exactly which product was attempted.
    if n != k and m > 0:
        raise ValueError(
            f"shape mismatch: {(m, n)} @ {(k, p)}; inner dimensions {n} and {k} differ"
        )

    # Entry (i, j) is row i of A dotted with column j of B. The shared index t runs over
    # the inner dimension, which is the only index that is summed away.
    result: list[list[float]] = []
    for i in range(m):
        row: list[float] = []
        for j in range(p):
            total: float = 0
            for t in range(n):
                total += A[i][t] * B[t][j]
            row.append(total)
        result.append(row)
    return result
