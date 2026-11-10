from __future__ import annotations


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Wrong on purpose: ints are converted to floats in the result."""
    # The empty list is the (0, 0) matrix, and its transpose is also empty.
    if not matrix:
        return []

    # A matrix is a list of rows, so every row must be a list or tuple. A flat list is a
    # vector, not a matrix: catch it here, before we ask for the length of a number.
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"expected a list of rows; row {i} is {type(row).__name__}")

    # Rectangular means every row has the length of row 0. zip(*rows) would silently
    # truncate ragged rows, so the lengths are checked explicitly.
    n_cols = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != n_cols:
            raise ValueError(f"ragged rows: row 0 has {n_cols} entries, row {i} has {len(row)}")

    # Row j of the result is column j of the input: collect entry j from every input row.
    # Each inner list is a new object, so the result never shares rows with the input.
    return [[float(matrix[i][j]) for i in range(len(matrix))] for j in range(n_cols)]
