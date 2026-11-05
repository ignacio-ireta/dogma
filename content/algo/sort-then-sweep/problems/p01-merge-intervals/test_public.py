"""Public tests: the examples from problem.md, touching and separate intervals, and the invalid
interval.

Each failure line names the mistake it guards against.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import pairwise
from typing import Any

import pytest
from submission import merge_intervals

SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one small call runs past SMALL_CALL_S."""


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


def _result(intervals: list[tuple[int, int]]) -> Any:
    """The learner's answer. A hang or a crash fails the test with the mistake named."""
    if _HUNG:
        pytest.fail(_HUNG[0])
    try:
        with _time_box(SMALL_CALL_S):
            return merge_intervals(intervals)
    except _TooSlow:
        message = (
            f"the call on {len(intervals)} intervals did not finish in {SMALL_CALL_S:g} s: sort "
            f"the "
            "intervals once, then make a single pass over them"
        )
        _HUNG.append(message)
        pytest.fail(message)
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on a small list: {exc}")


def _cells(intervals: Any) -> set[int]:
    """Doubled-grid cells covered by closed intervals: whole number x is cell 2x, the gap after x "
    "is 2x + 1."""
    cells: set[int] = set()
    for start, end in intervals:
        cells.update(range(2 * start, 2 * end + 1))
    return cells


def _explain(got: Any, want: list[tuple[int, int]], source: list[tuple[int, int]]) -> str | None:
    """Name the mistake behind a wrong answer, or return None. Checks run from the cheapest to "
    "the widest."""
    if not isinstance(got, list) or not all(type(item) is tuple and len(item) == 2 for item in got):
        return "return a list of (start, end) tuples: each merged interval must be a tuple"
    if got == want:
        return None
    for (_, first_end), (second_start, _) in pairwise(got):
        if second_start <= first_end:
            return (
                "two output intervals share a point: intervals that touch or overlap must merge; "
                "merge when "
                "start <= the last merged end"
            )
    if not all(_cells([item]) <= _cells(got) for item in source):
        return (
            "an input interval is not inside the output: sort by start first, and when two "
            "intervals merge keep "
            "the larger end (max), not the end of the newest interval"
        )
    if _cells(got) - _cells(source):
        return (
            "intervals that share no point were merged: a gap was bridged; merge only when start "
            "<= the last "
            "end, not last end + 1"
        )
    return (
        "the output pieces are not the maximal merged intervals: merge every touching or "
        "overlapping pair"
    )


def test_examples_from_problem_md() -> None:
    examples = [
        ([(5, 7), (1, 2), (2, 4), (9, 9)], [(1, 4), (5, 7), (9, 9)]),
        ([(3, 5), (1, 3)], [(1, 5)]),
        ([(1, 3), (4, 5)], [(1, 3), (4, 5)]),
        ([(0, 10), (2, 3), (4, 4)], [(0, 10)]),
        ([], []),
    ]
    for intervals, want in examples:
        problem = _explain(_result(intervals), want, intervals)
        assert problem is None, problem


def test_touching_intervals_merge_and_whole_number_neighbours_do_not() -> None:
    touching = _explain(_result([(1, 3), (3, 5)]), [(1, 5)], [(1, 3), (3, 5)])
    assert touching is None, (
        f"(1, 3) and (3, 5) share the point 3, so they must merge into (1, 5): {touching}"
    )
    apart = _explain(_result([(1, 3), (4, 5)]), [(1, 3), (4, 5)], [(1, 3), (4, 5)])
    assert apart is None, f"(1, 3) and (4, 5) share no point, so they stay two intervals: {apart}"


def test_invalid_interval_raises_value_error() -> None:
    try:
        merge_intervals([(1, 2), (5, 3)])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"an interval with start > end must raise ValueError, instead it raised "
            f"{type(exc).__name__}"
        )
    pytest.fail("an interval with start > end was accepted: raise ValueError when start > end")
