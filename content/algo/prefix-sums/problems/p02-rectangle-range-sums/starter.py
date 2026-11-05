from __future__ import annotations


def rectangle_sums(
    grid: list[list[int]], queries: list[tuple[int, int, int, int] | list[int]]
) -> list[int]:
    """Answer inclusive rectangle sums with one padded prefix table.

    Grid rows have equal length and contain Python ints, excluding bool. Each
    query is (top, left, bottom, right), all Python ints, with ordered inclusive
    bounds inside the grid. Empty grids may be queried only with an empty list.
    Raise TypeError for non-list grids/rows/query collections or non-int values;
    raise ValueError for ragged rows, wrong query lengths or invalid bounds.
    Query records may be tuples or lists. Never mutate inputs.
    Use O(rows*columns + len(queries)) arithmetic operations.
    """
    raise NotImplementedError
