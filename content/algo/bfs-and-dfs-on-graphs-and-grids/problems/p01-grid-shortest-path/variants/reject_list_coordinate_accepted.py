from __future__ import annotations

from collections import deque
from typing import Any

# Up, down, left, right. Every move costs 1, so BFS gives the fewest moves.
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _coordinate(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a (row, column) tuple")
    for item in value:
        # type(...) is int rejects bool, float and NumPy integers.
        if type(item) is not int:
            raise TypeError(f"{name} entries must be Python ints, got {type(item).__name__}")
    return value[0], value[1]


def _check_grid(grid: Any) -> tuple[int, int]:
    if type(grid) is not list:
        raise TypeError(f"grid must be a list of str, got {type(grid).__name__}")
    if not grid:
        raise ValueError("grid must have at least one row")
    width = None
    for row in grid:
        if type(row) is not str:
            raise TypeError(f"each grid row must be a str, got {type(row).__name__}")
        if width is None:
            width = len(row)
        if len(row) != width or width == 0:
            raise ValueError("grid rows must be non-empty and all the same length")
        if set(row) - {".", "#"}:
            raise ValueError("grid characters must be '.' (open) or '#' (wall)")
    return len(grid), width or 0


def shortest_path_length(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Wrong on purpose: a list coordinate is accepted where a tuple is required. Return the fewest moves from start to goal in a grid of open cells and walls, or -1.

    See starter.py for the full contract.
    """  # noqa: E501
    rows, cols = _check_grid(grid)
    start_row, start_col = _coordinate("start", start)
    goal_row, goal_col = _coordinate("goal", goal)
    for name, (row, col) in (("start", (start_row, start_col)), ("goal", (goal_row, goal_col))):
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"{name} lies outside the grid")
        if grid[row][col] == "#":
            raise ValueError(f"{name} is a wall")

    # distance[r][c] is -1 until the cell is reached. Marking happens when a cell is ENQUEUED, so
    # each open cell enters the queue at most once, and its distance is final from that moment.
    distance = [[-1] * cols for _ in range(rows)]
    distance[start_row][start_col] = 0
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    while queue:
        row, col = queue.popleft()  # FIFO: cells come out in order of distance from start
        if (row, col) == (goal_row, goal_col):
            return distance[row][col]
        for dr, dc in MOVES:
            nr, nc = row + dr, col + dc
            # Bounds first, so a negative index is never used to read the grid.
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == "#" or distance[nr][nc] != -1:
                continue
            distance[nr][nc] = distance[row][col] + 1
            queue.append((nr, nc))
    return -1
