from __future__ import annotations


def _shape(M: list, name: str) -> tuple[int, int]:
    if not isinstance(M, list):
        raise ValueError(f"{name} must be a list of rows")
    if len(M) == 0:
        return (0, 0)
    for row in M:
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{name} must be a list of rows, not a flat sequence")
    width = len(M[0])
    for i, row in enumerate(M):
        if len(row) != width:
            raise ValueError(
                f"ragged rows in {name}: row 0 has {width} entries, row {i} has {len(row)}"
            )
    return (len(M), width)


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Return the matrix product ``A @ B`` for two matrices given as lists of rows.

    Parameters
    ----------
    A : list of rows, shape (m, n). Each row is a list or tuple of numbers; ``[]`` is (0, 0).
    B : list of rows, shape (n, p) with the same conventions.

    Returns
    -------
    list of m rows, each of length p: result[i][j] = sum_t A[i][t] * B[t][j].

    Raises
    ------
    ValueError
        If the inner dimensions differ (the message must name both shapes, e.g.
        ``(2, 3) @ (2, 2)``), or if either matrix is ragged or not a list of rows.
    """
    m, n = _shape(A, "A")
    k, p = _shape(B, "B")
    if n != k:
        raise ValueError(
            f"shape mismatch: ({m}, {n}) @ ({k}, {p}); inner dimensions {n} and {k} differ"
        )

    is_float = any(isinstance(x, float) for row in (A + B) for x in row)
    zero = 0.0 if is_float else 0

    return [[sum((A[i][t] * B[t][j] for t in range(n)), zero) for j in range(p)] for i in range(m)]
