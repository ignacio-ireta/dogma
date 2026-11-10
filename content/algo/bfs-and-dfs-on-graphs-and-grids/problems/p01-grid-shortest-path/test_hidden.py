"""Hidden tests: a level-by-level frontier oracle on seeded grids, edge, validation and time rules.

Failure messages name the mistake. They never print hidden grids or expected answers.
A call that runs past its time box is stopped by a signal and reported by name. The previous
signal handler is restored after each call.
"""

from __future__ import annotations

import random
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from submission import shortest_path_length

SMALL_CALL_S = 5.0
LARGE_CALL_S = 10.0
LARGE_SIDE = 700
SLOW_MESSAGE = (
    "a search did not finish in time: a search without a visited record re-explores cells "
    "along every route; mark each cell when you first reach it"
)


class _TooSlow(BaseException):
    """Raised inside the learner's function when a call runs past its time box."""


_HUNG: list[str] = []


@contextmanager
def _time_box(seconds: float) -> Iterator[None]:
    """Interrupt the learner's code after ``seconds``. Only the main thread can use signals."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    def _expired(signum: int, frame: Any) -> None:
        raise _TooSlow

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _call(grid: list[str], start: Any, goal: Any, limit: float) -> tuple[Any, str | None]:
    """Return (answer, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(limit):
            return shortest_path_length(grid, start, goal), None
    except _TooSlow:
        _HUNG.append(SLOW_MESSAGE)
        return None, SLOW_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a valid grid: check the bounds and the "
            "neighbour loop"
        )


def _oracle(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Independent answer: expand whole frontiers level by level, using sets, not a queue."""
    rows, cols = len(grid), len(grid[0])
    seen = {start}
    frontier = {start}
    level = 0
    while frontier:
        if goal in frontier:
            return level
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr][nc] == "."
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    nxt.add((nr, nc))
        frontier = nxt
        level += 1
    return -1


def _check(grid: list[str], start: tuple[int, int], goal: tuple[int, int], limit: float) -> None:
    got, problem = _call(grid, start, goal, limit)
    if problem is not None:
        pytest.fail(problem)
    if type(got) is not int:
        pytest.fail("return a Python int number of moves, not a float or a NumPy scalar")
    expected = _oracle(grid, start, goal)
    if got != expected:
        pytest.fail(
            "the move count is wrong: moves are up, down, left and right with unit cost, and walls "
            "must never be entered"
        )


def _random_grid(rng: random.Random, rows: int, cols: int, wall_rate: float) -> list[str]:
    return [
        "".join("#" if rng.random() < wall_rate else "." for _ in range(cols)) for _ in range(rows)
    ]


def _open_cell(rng: random.Random, grid: list[str]) -> tuple[int, int]:
    cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "."]
    return rng.choice(cells)


def test_seeded_random_grids_match_the_frontier_oracle() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic grids, not security.
    for _ in range(150):
        rows, cols = rng.randint(1, 12), rng.randint(1, 12)
        grid = _random_grid(rng, rows, cols, rng.choice([0.0, 0.2, 0.35, 0.5]))
        open_cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "."]
        if not open_cells:
            continue
        start = rng.choice(open_cells)
        goal = rng.choice(open_cells)
        _check(grid, start, goal, SMALL_CALL_S)


def test_a_route_that_depth_first_search_takes_first_is_not_taken_as_shortest() -> None:
    grid = [
        "........",
        "#######.",
        "........",
        ".#######",
        "........",
    ]
    _check(grid, (0, 0), (4, 0), SMALL_CALL_S)


def test_goal_sealed_off_by_walls_returns_minus_one() -> None:
    grid = ["..#..", "..#..", "###..", "....."]
    got, problem = _call(grid, (0, 0), (0, 4), SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != -1:
        pytest.fail("an unreachable goal must return -1: the search must end when no route is left")


def test_single_row_and_single_column_grids() -> None:
    _check(["......"], (0, 0), (0, 5), SMALL_CALL_S)
    _check(["."] * 7, (0, 0), (6, 0), SMALL_CALL_S)
    _check(["..#.."], (0, 0), (0, 4), SMALL_CALL_S)


def test_one_by_one_grid_returns_zero() -> None:
    got, problem = _call(["."], (0, 0), (0, 0), SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != 0:
        pytest.fail("a single open cell that is both start and goal is 0 moves")


def test_corner_to_corner_on_open_grid_is_manhattan_distance() -> None:
    grid = ["." * 9 for _ in range(6)]
    got, problem = _call(grid, (0, 0), (5, 8), SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != 13:
        pytest.fail("on an open grid the answer is the Manhattan distance between the cells")


def test_walls_on_the_border_do_not_wrap_around() -> None:
    grid = ["#.#", "...", "#.#"]
    _check(grid, (1, 0), (1, 2), SMALL_CALL_S)
    grid2 = [".#", "#."]
    got, problem = _call(grid2, (0, 0), (1, 1), SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != -1:
        pytest.fail(
            "the search must not wrap around the grid edge: a negative index is not a neighbour"
        )


def test_large_open_grid_finishes_within_the_time_limit() -> None:
    grid = ["." * LARGE_SIDE for _ in range(LARGE_SIDE)]
    got, problem = _call(grid, (0, 0), (LARGE_SIDE - 1, LARGE_SIDE - 1), LARGE_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != 2 * (LARGE_SIDE - 1):
        pytest.fail(
            "wrong distance on the large open grid: check the visited record and the frontier"
        )


def test_large_serpentine_maze_finishes_within_the_time_limit() -> None:
    side = 301
    maze = []
    for r in range(side):
        if r % 2 == 0:
            maze.append("." * side)
        else:
            row = ["#"] * side
            # Alternate the single opening between the right and left edges, making one long snake.
            row[side - 1 if (r // 2) % 2 == 0 else 0] = "."
            maze.append("".join(row))
    _check_value(maze, (0, 0), (side - 1, 0), _oracle(maze, (0, 0), (side - 1, 0)), LARGE_CALL_S)


def _check_value(grid: list[str], start: Any, goal: Any, expected: int, limit: float) -> None:
    got, problem = _call(grid, start, goal, limit)
    if problem is not None:
        pytest.fail(problem)
    if got != expected:
        pytest.fail(
            "wrong distance on a long maze: the route must follow the corridors, not shortcut walls"
        )


def test_start_on_a_wall_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length(["#."], (0, 0), (0, 1)),
        "a start on a wall must raise ValueError",
    )


def test_start_outside_the_grid_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length(["..", ".."], (2, 0), (0, 0)),
        "a start outside the grid must raise ValueError",
    )


def test_negative_coordinate_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length(["..", ".."], (-1, 0), (0, 0)),
        "a negative coordinate is outside the grid and must raise ValueError",
    )


def test_ragged_grid_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length(["..", "..."], (0, 0), (0, 1)),
        "rows of different lengths must raise ValueError",
    )


def test_unknown_character_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length([".x", ".."], (0, 0), (0, 1)),
        "a character other than '.' or '#' must raise ValueError",
    )


def test_empty_grid_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length([], (0, 0), (0, 0)),
        "an empty grid must raise ValueError",
    )


def test_non_str_row_raises_type_error() -> None:
    _expect(
        TypeError,
        lambda: shortest_path_length(["..", ["."]], (0, 0), (0, 1)),  # type: ignore[list-item]
        "a grid row that is not a str must raise TypeError",
    )


def test_bool_or_list_coordinates_raise_type_error() -> None:
    _expect(
        TypeError,
        lambda: shortest_path_length([".."], (True, 0), (0, 1)),  # type: ignore[arg-type]
        "a bool coordinate must raise TypeError, not be read as 1",
    )
    _expect(
        TypeError,
        lambda: shortest_path_length([".."], [0, 0], (0, 1)),  # type: ignore[arg-type]
        "a list start must raise TypeError: coordinates are a tuple of two ints",
    )


def _expect(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_grid_that_is_not_a_list_raises_type_error() -> None:
    _expect(
        TypeError,
        lambda: shortest_path_length(("..", ".."), (0, 0), (0, 1)),  # type: ignore[arg-type]
        "a grid that is not a list must raise TypeError",
    )


def test_negative_column_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: shortest_path_length(["..", ".."], (0, -1), (0, 0)),
        "a start with a negative column is outside the grid and must raise ValueError",
    )
    _expect(
        ValueError,
        lambda: shortest_path_length(["..", ".."], (0, 0), (1, -1)),
        "a goal with a negative column is outside the grid and must raise ValueError",
    )


def test_wall_cell_that_is_start_and_goal_raises_value_error() -> None:
    # Both cells must be open. Here they are the same cell, and it is a wall.
    _expect(
        ValueError,
        lambda: shortest_path_length(["#.."], (0, 0), (0, 0)),
        "a start that equals the goal on a wall must raise ValueError, not return a distance",
    )
