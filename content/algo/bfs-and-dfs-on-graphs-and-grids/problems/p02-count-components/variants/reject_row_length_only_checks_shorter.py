from __future__ import annotations

from typing import Any

# Up, down, left, right. Diagonal cells are not neighbours here.
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
        if len(row) < width or width == 0:
            raise ValueError("grid rows must be non-empty and all the same length")
        if set(row) - {".", "#"}:
            raise ValueError("grid characters must be '.' (open) or '#' (wall)")
    return len(grid), width or 0


def count_components(grid: list[str]) -> int:
    """Wrong on purpose: rows longer than the first row are accepted. Return the number of connected components of open cells, with 4-neighbour connectivity.

    See starter.py for the full contract.
    """  # noqa: E501
    rows, cols = _check_grid(grid)
    seen = [[False] * cols for _ in range(rows)]
    components = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#" or seen[r][c]:
                continue
            # An unseen open cell starts a new component. Flood the whole component with an
            # explicit stack, so a long snake-shaped island needs no deep recursion.
            components += 1
            seen[r][c] = True
            stack = [(r, c)]
            while stack:
                row, col = stack.pop()
                for dr, dc in MOVES:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if grid[nr][nc] == "#" or seen[nr][nc]:
                        continue
                    seen[nr][nc] = True
                    stack.append((nr, nc))
    return components
