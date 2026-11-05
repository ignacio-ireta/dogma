from __future__ import annotations


def count_components(grid: list[str]) -> int:
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
    raise NotImplementedError
