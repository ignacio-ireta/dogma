"""Public tests: the examples from problem.md, including diagonal cells and the all-wall grid."""

from __future__ import annotations

import pytest
from submission import count_components


def _count(grid):
    try:
        return count_components(grid)
    except IndexError:
        pytest.fail(
            "a neighbour index left the grid: check row and column bounds before indexing "
            "the grid or visited array",
            pytrace=False,
        )


def test_two_islands_with_one_bent_corner() -> None:
    assert _count(["..#", "..#", "##."]) == 2, (
        "wrong count for two islands: a 2x2 block is one component, the lone corner cell is another"
    )


def test_diagonal_cells_do_not_connect() -> None:
    assert _count([".#.", "#.#", ".#."]) == 5, (
        "diagonal neighbours are not connected: only up, down, left and right moves join cells"
    )


def test_all_walls_have_no_components() -> None:
    assert _count(["###", "###"]) == 0, "a grid of walls has no open cells, so 0 components"
