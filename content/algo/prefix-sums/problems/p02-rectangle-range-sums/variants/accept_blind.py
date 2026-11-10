from __future__ import annotations


def _is_int(value: object) -> bool:
    return type(value) is int


def rectangle_sums(  # noqa: PLR0912
    grid: list[list[int]], queries: list[tuple[int, int, int, int] | list[int]]
) -> list[int]:
    """Answer inclusive rectangle sums with one padded prefix table."""
    if type(grid) is not list:
        raise TypeError("grid must be a list of lists")
    if type(queries) is not list:
        raise TypeError("queries must be a list")

    rows = len(grid)
    columns = 0
    for index, row in enumerate(grid):
        if type(row) is not list:
            raise TypeError("every grid row must be a list")
        if index == 0:
            columns = len(row)
        elif len(row) != columns:
            raise ValueError("grid rows must all have the same length")
        for value in row:
            if not _is_int(value):
                raise TypeError("grid values must be Python ints")

    checked: list[tuple[int, int, int, int]] = []
    for query in queries:
        if type(query) is not tuple and type(query) is not list:
            raise TypeError("each query must be a tuple or list")
        for part in query:
            if not _is_int(part):
                raise TypeError("query bounds must be Python ints")
        if len(query) != 4:
            raise ValueError("each query must hold exactly four bounds")
        top, left, bottom, right = query
        if rows == 0 or columns == 0:
            raise ValueError("an empty grid has no valid rectangles")
        if not (0 <= top <= bottom < rows and 0 <= left <= right < columns):
            raise ValueError("query bounds are outside the grid or reversed")
        checked.append((top, left, bottom, right))

    if not checked:
        return []

    prefix = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows):
        running = 0
        row = grid[i]
        for j in range(columns):
            running += row[j]
            prefix[i + 1][j + 1] = prefix[i][j + 1] + running

    results: list[int] = []
    for top, left, bottom, right in checked:
        results.append(
            prefix[bottom + 1][right + 1]
            - prefix[top][right + 1]
            - prefix[bottom + 1][left]
            + prefix[top][left]
        )
    return results
