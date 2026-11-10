from __future__ import annotations


def count_components(grid: list[str]) -> int:  # noqa: PLR0912
    """Return the number of connected components of open cells, with 4-neighbour connectivity.

    Parameters
    ----------
    grid : non-empty list of str, equal-length rows, each character '.' (open) or '#' (wall).
        Not changed.

    Returns
    -------
    int
        The number of components. 0 when there are no open cells.

    Raises
    ------
    TypeError
        When grid is not a list of str.
    ValueError
        When the grid is empty or ragged, or holds a character other than '.' or '#'.
    """
    if not isinstance(grid, list):
        raise TypeError("grid must be a list of str")
    if len(grid) == 0:
        raise ValueError("grid must not be empty")
    for row in grid:
        if not isinstance(row, str):
            raise TypeError("every row must be a str")
    width = len(grid[0])
    if width == 0:
        raise ValueError("rows must not be empty")
    for row in grid:
        if len(row) != width:
            raise ValueError("rows must all have the same length")
        for ch in row:
            if ch != "." and ch != "#":  # noqa: PLR1714
                raise ValueError("grid characters must be '.' or '#'")

    height = len(grid)
    seen = [[False] * width for _ in range(height)]
    count = 0
    for r in range(height):
        for c in range(width):
            if grid[r][c] != "." or seen[r][c]:
                continue
            count += 1
            seen[r][c] = True
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr][nc]
                        and grid[nr][nc] == "."
                    ):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
    return count
