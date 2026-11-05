from __future__ import annotations


def shortest_path_length(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> int:
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
    raise NotImplementedError
