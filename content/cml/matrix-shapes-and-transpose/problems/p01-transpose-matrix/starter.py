from __future__ import annotations


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Return the transpose of ``matrix`` as a new list of lists.

    Parameters
    ----------
    matrix : list of rows, each row a list or tuple of numbers. Shape (r, c) means r rows of
        length c; ``[]`` is the (0, 0) matrix.

    Returns
    -------
    list of lists with shape (c, r): result[j][i] == matrix[i][j]. Zero-width input gives [].

    Raises
    ------
    ValueError
        If the rows have different lengths, or ``matrix`` is not a list of rows.
    """
    raise NotImplementedError
