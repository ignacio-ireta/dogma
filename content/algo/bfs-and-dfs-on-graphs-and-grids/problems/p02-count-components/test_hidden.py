"""Hidden tests: a union-find oracle on seeded grids, long snakes, borders, types and time limits.

Failure messages name the mistake. They never print hidden grids or expected counts.
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
from submission import count_components

SMALL_CALL_S = 5.0
LARGE_CALL_S = 10.0
SNAKE_SIDE = 301
SLOW_MESSAGE = (
    "a count did not finish in time: each cell must be visited once, and each island flooded once"
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


def _call(grid: list[str], limit: float) -> tuple[Any, str | None]:
    """Return (count, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(limit):
            return count_components(grid), None
    except _TooSlow:
        _HUNG.append(SLOW_MESSAGE)
        return None, SLOW_MESSAGE
    except RecursionError:
        return None, (
            "the search ran past Python's recursion limit on one long island: use an explicit "
            "stack or queue instead of recursion"
        )
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a valid grid: check the bounds and the "
            "neighbour loop"
        )


def _oracle(grid: list[str]) -> int:
    """Independent answer: union-find over open cells joined to their right and lower neighbours."""
    rows, cols = len(grid), len(grid[0])
    parent = list(range(rows * cols))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    open_count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != ".":
                continue
            open_count += 1
            for nr, nc in ((r, c + 1), (r + 1, c)):
                if nr < rows and nc < cols and grid[nr][nc] == ".":
                    a, b = find(r * cols + c), find(nr * cols + nc)
                    if a != b:
                        parent[a] = b
    roots = {find(r * cols + c) for r in range(rows) for c in range(cols) if grid[r][c] == "."}
    return len(roots) if open_count else 0


def _check(grid: list[str], limit: float = SMALL_CALL_S) -> None:
    got, problem = _call(grid, limit)
    if problem is not None:
        pytest.fail(problem)
    if type(got) is not int:
        pytest.fail("return a Python int count, not a float or another numeric type")
    expected = _oracle(grid)
    if got != expected:
        pytest.fail(
            "the component count is wrong: join cells only through up, down, left and right, and "
            "count each island once"
        )


def _random_grid(rng: random.Random, rows: int, cols: int, wall_rate: float) -> list[str]:
    return [
        "".join("#" if rng.random() < wall_rate else "." for _ in range(cols)) for _ in range(rows)
    ]


def test_seeded_random_grids_match_the_union_find_oracle() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic grids, not security.
    for _ in range(150):
        rows, cols = rng.randint(1, 14), rng.randint(1, 14)
        _check(_random_grid(rng, rows, cols, rng.choice([0.0, 0.3, 0.45, 0.6])))


def test_all_open_grid_is_one_component() -> None:
    _check(["." * 9 for _ in range(7)])


def test_all_wall_grid_has_zero_components() -> None:
    got, problem = _call(["#" * 4 for _ in range(3)], SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != 0:
        pytest.fail("a grid of walls must give 0 components")


def test_single_row_and_single_column() -> None:
    _check(["..#.."])
    _check(["."] * 6)
    _check(["."])


def test_diagonal_touching_islands_stay_separate() -> None:
    _check(["..#", "#..", "..#"])


def test_islands_touching_the_border_are_counted_once() -> None:
    _check([".#.", "...", ".#."])


def test_long_snake_island_does_not_hit_the_recursion_limit() -> None:
    maze = []
    for r in range(SNAKE_SIDE):
        if r % 2 == 0:
            maze.append("." * SNAKE_SIDE)
        else:
            row = ["#"] * SNAKE_SIDE
            row[SNAKE_SIDE - 1 if (r // 2) % 2 == 0 else 0] = "."
            maze.append("".join(row))
    got, problem = _call(maze, LARGE_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != 1:
        pytest.fail("a single snake-shaped island is one component: flood it completely")


def test_large_grid_of_many_small_islands_finishes_within_the_time_limit() -> None:
    rng = random.Random(7)  # noqa: S311 -- seeded synthetic grid, not security.
    grid = _random_grid(rng, 400, 400, 0.55)
    got, problem = _call(grid, LARGE_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if got != _oracle(grid):
        pytest.fail(
            "the component count is wrong on the large grid: check the seen marks and the flood"
        )


def _expect(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_empty_grid_raises_value_error() -> None:
    _expect(ValueError, lambda: count_components([]), "an empty grid must raise ValueError")


def test_ragged_grid_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: count_components(["..", "..."]),
        "rows of different lengths must raise ValueError",
    )


def test_unknown_character_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: count_components([".x"]),
        "a character other than '.' or '#' must raise ValueError",
    )


def test_non_str_row_raises_type_error() -> None:
    _expect(
        TypeError,
        lambda: count_components(["..", ["."]]),  # type: ignore[list-item]
        "a grid row that is not a str must raise TypeError",
    )


def test_grid_that_is_not_a_list_raises_type_error() -> None:
    _expect(
        TypeError,
        lambda: count_components(("..", "..")),  # type: ignore[arg-type]
        "a grid that is not a list must raise TypeError",
    )
