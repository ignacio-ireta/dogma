from __future__ import annotations

import pytest
from submission import rectangle_sums


def test_inclusive_rectangles_match_the_worked_table():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert rectangle_sums(grid, [(1, 1, 2, 2), (0, 0, 2, 2), (1, 1, 1, 1)]) == [28, 45, 5]


def test_negative_values_and_repeated_queries():
    grid = [[2, -3], [-1, 4]]
    assert rectangle_sums(grid, [(0, 0, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1)]) == [2, 1, 2]


def test_empty_query_collection_and_reversed_bounds():
    assert rectangle_sums([], []) == []
    with pytest.raises(ValueError):
        rectangle_sums([[1, 2]], [(0, 1, 0, 0)])
