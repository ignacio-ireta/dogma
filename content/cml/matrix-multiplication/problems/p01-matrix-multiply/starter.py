from __future__ import annotations


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
    raise NotImplementedError
