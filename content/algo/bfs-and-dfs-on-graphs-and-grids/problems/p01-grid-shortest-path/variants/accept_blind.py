from __future__ import annotations

from collections import deque


def _check_point(name: str, point: object) -> tuple[int, int]:
    if not isinstance(point, tuple):
        raise TypeError(f"{name} must be a tuple of two ints")
    if len(point) != 2:
        raise ValueError(f"{name} must have exactly two coordinates")
    for coord in point:
        if isinstance(coord, bool) or not isinstance(coord, int):
            raise TypeError(f"{name} coordinates must be Python ints")
    return point[0], point[1]


def shortest_path_length(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> int:  # noqa: PLR0912
    """Return the fewest moves from start to goal in a grid of open cells and walls, or -1.

    Parameters
    ----------
    grid : non-empty list of str, equal-length rows, each character '.' (open) or '#' (wall).
        Row 0 is the top row. Not changed.
    start : tuple of two Python int, (row, column), inside the grid and open.
    goal : tuple of two Python int, (row, column), inside the grid and open.

    Returns
    -------
    int
        0 when start equals goal, the minimum number of up, down, left or right moves otherwise,
        and -1 when goal cannot be reached.

    Raises
    ------
    TypeError
        When grid is not a list of str, or a coordinate is not a Python int (bool included).
    ValueError
        When the grid is empty or ragged, holds a character other than '.' or '#', or a cell is
        outside the grid or a wall.
    """
    if not isinstance(grid, list):
        raise TypeError("grid must be a list of str")
    for row in grid:
        if not isinstance(row, str):
            raise TypeError("every row of grid must be a str")
    if len(grid) == 0:
        raise ValueError("grid must not be empty")
    width = len(grid[0])
    if width == 0:
        raise ValueError("grid rows must not be empty")
    for row in grid:
        if len(row) != width:
            raise ValueError("grid rows must all have the same length")
        for ch in row:
            if ch not in ".#":
                raise ValueError(f"grid characters must be '.' or '#', got {ch!r}")

    start = _check_point("start", start)
    goal = _check_point("goal", goal)
    height = len(grid)
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not (0 <= r < height and 0 <= c < width):
            raise ValueError(f"{name} lies outside the grid")
        if grid[r][c] != ".":
            raise ValueError(f"{name} must be an open cell")

    if start == goal:
        return 0

    seen = bytearray(height * width)
    seen[start[0] * width + start[1]] = 1
    queue: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])
    while queue:
        r, c, dist = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            idx = nr * width + nc
            if seen[idx] or grid[nr][nc] != ".":
                continue
            if (nr, nc) == goal:
                return dist + 1
            seen[idx] = 1
            queue.append((nr, nc, dist + 1))
    return -1
