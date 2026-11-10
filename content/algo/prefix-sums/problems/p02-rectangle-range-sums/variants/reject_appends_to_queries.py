from __future__ import annotations


def rectangle_sums(
    grid: list[list[int]], queries: list[tuple[int, int, int, int] | list[int]]
) -> list[int]:
    """Wrong on purpose: appends a query to the caller's list. Answer inclusive rectangle sums with one padded prefix table.

    Grid rows have equal length and contain Python ints, excluding bool. Each
    query is (top, left, bottom, right), all Python ints, with ordered inclusive
    bounds inside the grid. Empty grids may be queried only with an empty list.
    Raise TypeError for non-list grids/rows/query collections or non-int values;
    raise ValueError for ragged rows, wrong query lengths or invalid bounds.
    Query records may be tuples or lists. Never mutate inputs.
    Use O(rows*columns + len(queries)) arithmetic operations.
    """  # noqa: E501
    if not isinstance(grid, list) or not isinstance(queries, list):
        raise TypeError("grid and queries must be lists")
    rows = len(grid)
    if any(not isinstance(row, list) for row in grid):
        raise TypeError("each grid row must be a list")
    columns = len(grid[0]) if rows else 0
    if any(len(row) != columns for row in grid):
        raise ValueError("grid rows must have equal length")
    if any(type(value) is not int for row in grid for value in row):
        raise TypeError("grid entries must be Python integers")
    for query in queries:
        if not isinstance(query, tuple | list):
            raise TypeError("each query must be a tuple or list")
        if len(query) != 4:
            raise ValueError("each query needs four coordinates")
        if any(type(value) is not int for value in query):
            raise TypeError("query coordinates must be Python integers")
        top, left, bottom, right = query
        if not (0 <= top <= bottom < rows and 0 <= left <= right < columns):
            raise ValueError("rectangle bounds must be ordered and inside the grid")
    if not queries:
        return []

    # Zero padding makes origin-edge queries use the same four-corner identity.
    prefix = [[0] * (columns + 1) for _ in range(rows + 1)]
    for r, row in enumerate(grid, start=1):
        row_sum = 0
        for c, value in enumerate(row, start=1):
            row_sum += value
            prefix[r][c] = prefix[r - 1][c] + row_sum

    # Inclusive bottom/right coordinates become exclusive padded table indices.
    answers = [
        prefix[bottom + 1][right + 1]
        - prefix[top][right + 1]
        - prefix[bottom + 1][left]
        + prefix[top][left]
        for top, left, bottom, right in queries
    ]
    queries.append(queries[0])
    return answers
