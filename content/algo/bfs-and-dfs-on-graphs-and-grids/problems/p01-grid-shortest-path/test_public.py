"""Public tests: the examples from problem.md, a start-equals-goal case and a wall-goal error."""

from __future__ import annotations

import pytest
from submission import shortest_path_length


def test_detour_around_a_wall_has_four_moves() -> None:
    got = shortest_path_length(["...", "##.", "..."], (0, 0), (2, 2))
    assert got == 4, (
        "wrong length around the wall: count moves, and go around the wall, not through it"
    )


def test_start_sealed_by_walls_returns_minus_one() -> None:
    got = shortest_path_length([".#.", "##.", "..."], (0, 0), (2, 2))
    assert got == -1, "an unreachable goal must return -1, not a distance"


def test_start_equal_to_goal_returns_zero() -> None:
    assert shortest_path_length(["..", ".."], (1, 1), (1, 1)) == 0, "start equal to goal is 0 moves"


def test_wall_goal_raises_value_error() -> None:
    try:
        shortest_path_length(["..#", "#.."], (0, 0), (0, 2))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a wall goal must raise ValueError, instead it raised {type(exc).__name__}")
    pytest.fail("a wall goal must raise ValueError: the goal cell is not open")
