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
    if not isinstance(matrix, list):
        raise ValueError("matrix must be a list of rows")
    if len(matrix) == 0:
        return []
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            raise ValueError("matrix must be a list of rows, not a flat sequence")
    width = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != width:
            raise ValueError(f"ragged rows: row 0 has {width} entries, row {i} has {len(row)}")
    if width == 0:
        return []
    return [[row[j] for row in matrix] for j in range(width)]
