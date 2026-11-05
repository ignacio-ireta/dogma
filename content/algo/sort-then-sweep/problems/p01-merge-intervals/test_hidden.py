"""Hidden tests: a coverage oracle on small random lists, edge families, invalid input, no
mutation, and a time limit.

The oracle marks coverage on a half-integer grid: each whole number and each open gap between two
neighbours
gets its own cell. Two intervals then share a point exactly when their cells are contiguous, so
touching
intervals merge and intervals one whole number apart do not. The 200,000-interval test runs the
learner's
function in a child process that is killed at its limit. Each failure line names the likely mistake
first.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import merge_intervals

LARGE_N = 200_000
TIME_LIMIT_S = 5.0
SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one small call runs past SMALL_CALL_S."""


HANG_MESSAGE = (
    "a call on a small list did not finish: sort the intervals once and make one pass; a loop that "
    "repeats until nothing changes can run forever"
)
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


def _fail(why: str) -> None:
    """The one place these tests raise, so a Hypothesis run reports a single named failure."""
    raise AssertionError(why)


def _call(intervals: list[tuple[int, int]]) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(SMALL_CALL_S):
            return merge_intervals(intervals), None
    except _TooSlow:
        _HUNG.append(HANG_MESSAGE)
        return None, HANG_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a small list: check the empty list, a "
            f"single "
            "interval, and negative endpoints"
        )


def _brute(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Independent answer: mark every grid cell covered, then read off the runs of covered cells."""
    if not intervals:
        return []
    low = min(start for start, _ in intervals)
    high = max(end for _, end in intervals)
    # Cell 2 * (x - low) is the whole number x; cell 2 * (x - low) + 1 is the open gap after x.
    covered = [False] * (2 * (high - low) + 1)
    for start, end in intervals:
        for cell in range(2 * (start - low), 2 * (end - low) + 1):
            covered[cell] = True
    runs: list[tuple[int, int]] = []
    cell = 0
    while cell < len(covered):
        if covered[cell]:
            first = cell
            while cell + 1 < len(covered) and covered[cell + 1]:
                cell += 1
            # A run starts and ends on a whole-number cell, so halving the cell gives the endpoints.
            runs.append((low + first // 2, low + cell // 2))
        cell += 1
    return runs


def _cells(intervals: Sequence[tuple[int, int]]) -> set[int]:
    """Doubled-grid cells covered by closed intervals: whole number x is cell 2x, the gap after x "
    "is 2x + 1."""
    cells: set[int] = set()
    for start, end in intervals:
        cells.update(range(2 * start, 2 * end + 1))
    return cells


def _first_problem(  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    got: Any, want: list[tuple[int, int]], source: list[tuple[int, int]]
) -> str | None:
    """Return the first mistake found, or None. Checks run from the cheapest to the widest."""
    if not isinstance(got, list) or not all(type(item) is tuple and len(item) == 2 for item in got):
        return "return a list of (start, end) tuples: each merged interval must be a tuple"
    if not all(type(value) is int for item in got for value in item):
        return "the endpoints must be plain Python ints"
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


def _verify(intervals: list[tuple[int, int]]) -> None:
    want = _brute(
        intervals
    )  # computed before the learner runs, so an edit to the input cannot skew it
    got, problem = _call(intervals)
    if problem is None:
        problem = _first_problem(got, want, intervals)
    if problem is not None:
        _fail(problem)


def _random_intervals(
    rng: np.random.Generator, n: int, low: int, high: int
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for _ in range(n):
        start = int(rng.integers(low, high))
        length = int(rng.integers(0, 6))
        intervals.append((start, start + length))
    return intervals


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_intervals_match_the_coverage_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    intervals = _random_intervals(rng, int(rng.integers(0, 12)), low=-8, high=20)
    _verify(intervals)


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_medium_random_intervals_match_the_coverage_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    intervals = _random_intervals(rng, int(rng.integers(40, 150)), low=0, high=300)
    _verify(intervals)


def test_empty_and_single_interval() -> None:
    _verify([])
    _verify([(4, 9)])


def test_duplicate_and_point_intervals_collapse() -> None:
    values = [(2, 2), (2, 2), (7, 7), (2, 2)]
    got, problem = _call(values)
    if problem is None and got != [(2, 2), (7, 7)]:
        problem = (
            "duplicate or single-point intervals were not collapsed: equal intervals "
            "share every point"
        )
    if problem is not None:
        _fail(problem)


def test_nested_interval_does_not_shrink_the_range() -> None:
    got, problem = _call([(0, 10), (2, 3), (4, 4)])
    if problem is None and got != [(0, 10)]:
        problem = (
            "a nested interval shrank the merged range: keep the larger of the two ends (max), "
            "not the "
            "end of the newest interval"
        )
    if problem is not None:
        _fail(problem)


def test_chain_of_touching_intervals_becomes_one() -> None:
    got, problem = _call([(2, 3), (0, 1), (1, 2), (3, 4)])
    if problem is None and got != [(0, 4)]:
        problem = (
            "a chain of touching intervals did not merge into one: each touches the previous one"
        )
    if problem is not None:
        _fail(problem)


def test_intervals_one_whole_number_apart_stay_apart() -> None:
    got, problem = _call([(4, 5), (1, 3), (6, 6)])
    if problem is None and got != [(1, 3), (4, 5), (6, 6)]:
        problem = "a gap was bridged: compare start with last end without adding one"
    if problem is not None:
        _fail(problem)


def test_unsorted_input_with_overlaps_is_sorted_first() -> None:
    got, problem = _call([(10, 12), (1, 4), (9, 11), (3, 5), (0, 0)])
    if problem is None and got != [(0, 0), (1, 5), (9, 12)]:
        problem = "intervals were merged in input order: sort by start before the single pass"
    if problem is not None:
        _fail(problem)


def test_negative_endpoints_are_ordered_as_numbers() -> None:
    got, problem = _call([(-3, -1), (-1, 2), (-10, -8)])
    if problem is None and got != [(-10, -8), (-3, 2)]:
        problem = (
            "negative endpoints were ordered or merged wrongly: compare them as the "
            "numbers they are"
        )
    if problem is not None:
        _fail(problem)


def test_long_interval_that_starts_first_absorbs_the_later_ones() -> None:
    # Sorting by end would put (1, 2) and (5, 6) before (0, 10), and then (5, 10) would appear.
    got, problem = _call([(5, 6), (0, 10), (1, 2)])
    if problem is None and got != [(0, 10)]:
        problem = (
            "a long interval that starts first must absorb the intervals inside it: sort by start"
        )
    if problem is not None:
        _fail(problem)


def _must_raise_value_error(intervals: list[tuple[int, int]], mistake: str) -> None:
    try:
        with _time_box(SMALL_CALL_S):
            merge_intervals(intervals)
    except ValueError:
        return
    except _TooSlow:
        _fail(f"{mistake}; instead the call did not finish")
    except Exception as exc:
        _fail(f"{mistake}; instead it raised {type(exc).__name__}")
    _fail(mistake)


def test_invalid_interval_raises_value_error() -> None:
    _must_raise_value_error(
        [(1, 2), (5, 3)],
        "an interval with start > end was accepted: raise ValueError when start > end",
    )


def test_invalid_interval_anywhere_in_the_list_raises_value_error() -> None:
    _must_raise_value_error(
        [(0, 1), (2, 3), (9, 4), (6, 7)],
        "an invalid interval in the middle of the list was not reported: validate every interval "
        "first",
    )


def test_input_list_and_tuples_are_not_mutated() -> None:
    intervals = [(9, 12), (1, 4), (3, 5)]
    before = list(intervals)
    _, problem = _call(intervals)
    if problem is not None:
        _fail(problem)
    if intervals != before:
        _fail("the input list was changed: sort a new list (sorted) instead of sorting in place")


_CHILD = f"""
import random
from submission import merge_intervals
n = {LARGE_N}
intervals = [(i, i + 1) for i in range(n)]
random.Random(7).shuffle(intervals)
try:
    res = merge_intervals(intervals)
except Exception as exc:
    print("RAISED", type(exc).__name__)
else:
    first = res[0] if res else (None, None)
    print("RESULT", len(res), first[0], first[1], res[-1][1] if res else None)
"""


def test_large_list_of_200000_intervals_finishes_within_the_time_limit() -> None:
    # Analytic answer: each interval touches the next one, so everything merges into (0, n).
    n = LARGE_N
    run_dir = Path(__file__).resolve().parent
    try:
        # S603: runs this interpreter on a fixed child script, never on learner-supplied input.
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _CHILD],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"too slow on {n} intervals: the function did not finish in {TIME_LIMIT_S:g} s. "
            f"Comparing every "
            "pair, or repeating a pass until nothing changes, is quadratic; sort once, then one "
            "pass"
        )
    lines = proc.stdout.strip().splitlines()
    line = lines[-1] if lines else ""
    if line.startswith("RAISED"):
        pytest.fail(f"your function raised on the large input: {line[len('RAISED ') :]}")
    fields = line.split()
    if not fields or fields[0] != "RESULT":
        pytest.fail("large run returned no result: check runtime errors and return type")
    count, first_start, first_end, last_end = fields[1:]
    if count != "1":
        _fail("the chain of touching intervals must merge into a single interval")
    if first_start != "0" or first_end != str(n) or last_end != str(n):
        _fail("the merged interval has wrong endpoints: preserve the minimum start and maximum end")
