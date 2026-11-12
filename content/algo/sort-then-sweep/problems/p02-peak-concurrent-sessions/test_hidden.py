"""Hidden tests: a brute-force count at every event time, edge families, invalid input, no
mutation, a time limit.

The oracle counts, at every event time, the sessions that contain that moment. It is quadratic, so
it runs on
small and medium lists. The 200,000-session test runs the learner's function in a child process
killed at its
limit. Each failure line names the likely mistake first.
"""

from __future__ import annotations

import functools
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import max_concurrent

from dogma_data import retail_events

LARGE_N = 200_000
TIME_LIMIT_S = 5.0
SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one small call runs past SMALL_CALL_S."""


HANG_MESSAGE = (
    "a call on a small list did not finish: sort the events once and make one pass with "
    "a running count"
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


def _call(sessions: list[tuple[int, int]]) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(SMALL_CALL_S):
            return max_concurrent(sessions), None
    except _TooSlow:
        _HUNG.append(HANG_MESSAGE)
        return None, HANG_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a small list: check the empty list and "
            f"the "
            "shared endpoints"
        )


def _brute(sessions: Sequence[tuple[int, int]]) -> int:
    """Independent answer: at every event time, count the sessions that contain that moment."""
    if not sessions:
        return 0
    moments = sorted({time for session in sessions for time in session})
    return max(sum(1 for start, end in sessions if start <= t < end) for t in moments)


def _first_problem(got: Any, want: int) -> str | None:
    if isinstance(got, bool) or not isinstance(got, int):
        return "return the peak as a plain int, not a float or a NumPy scalar"
    if got == want:
        return None
    if got > want:
        return (
            "the peak is too high: touching sessions were counted as overlapping, or a session "
            "that has ended "
            "was still counted; at a shared moment apply the end before the start"
        )
    return (
        "the peak is too low: count every session during its half-open span [start, end), and "
        "keep the largest count"
    )


def _verify(sessions: list[tuple[int, int]]) -> None:
    want = _brute(
        sessions
    )  # computed before the learner runs, so an edit to the input cannot skew it
    got, problem = _call(sessions)
    if problem is None:
        problem = _first_problem(got, want)
    if problem is not None:
        _fail(problem)


@functools.cache
def _real_spans() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Real intervals from the invoices of an online retailer (online_retail).

    The first set holds one interval per trading day: the hours from the day's first invoice to
    its last, counted from noon, so mornings are negative. The second holds one interval per
    customer and week: the hours of the week from that customer's first invoice to the last.
    Both are whole numbers, many intervals are a single point, and many share an endpoint.
    """
    events = retail_events()
    hours = events["ts"].dt.hour
    days = hours.groupby(events["ts"].dt.date).agg(["min", "max"])
    week_hour = events["ts"].dt.dayofweek * 24 + hours
    weeks = week_hour.groupby([events["user_id"], events["ts"].dt.strftime("%G-%V")]).agg(
        ["min", "max"]
    )
    return (
        tuple((int(a) - 12, int(b) - 12) for a, b in zip(days["min"], days["max"], strict=True)),
        tuple((int(a), int(b)) for a, b in zip(weeks["min"], weeks["max"], strict=True)),
    )


def _random_sessions(
    rng: np.random.Generator, n: int, low: int, high: int
) -> list[tuple[int, int]]:
    """n real sessions: daily trading windows for a small draw, customer weeks for a larger one.

    A session ends one hour after its last invoice, so every session has a length of at least
    1 and endpoints often coincide. The bounds are not used.
    """
    by_day, by_week = _real_spans()
    pool = by_day if n < 40 else by_week
    picked = [pool[int(i)] for i in rng.integers(0, len(pool), size=n)]
    return [(start, end + 1) for start, end in picked]


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_sessions_match_the_brute_force_count(seed: int) -> None:
    rng = np.random.default_rng(seed)
    _verify(_random_sessions(rng, int(rng.integers(0, 14)), low=0, high=16))


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_medium_random_sessions_match_the_brute_force_count(seed: int) -> None:
    rng = np.random.default_rng(seed)
    _verify(_random_sessions(rng, int(rng.integers(40, 160)), low=0, high=120))


def test_empty_and_single_session() -> None:
    _verify([])
    _verify([(4, 9)])


def test_touching_sessions_do_not_overlap_in_either_input_order() -> None:
    for sessions in ([(1, 3), (3, 5)], [(3, 5), (1, 3)]):
        got, problem = _call(list(sessions))
        if problem is None and got != 1:
            problem = (
                "the answer depends on the order of the input: at a shared moment the end must be "
                "applied "
                "before the start, whatever order the sessions were listed in"
            )
        if problem is not None:
            _fail(problem)


def test_identical_sessions_all_overlap() -> None:
    got, problem = _call([(4, 9)] * 5)
    if problem is None and got != 5:
        problem = (
            "identical sessions must all count at the same moments: the peak is the "
            "number of copies"
        )
    if problem is not None:
        _fail(problem)


def test_nested_sessions_peak_at_the_inner_overlap() -> None:
    got, problem = _call([(0, 100), (10, 20), (30, 40)])
    if problem is None and got != 2:
        problem = "nested sessions: keep the outer session active while counting each inner overlap"
    if problem is not None:
        _fail(problem)


def test_many_sessions_end_and_many_start_at_one_moment() -> None:
    sessions = [(0, 5), (0, 5), (0, 5), (5, 8), (5, 8)]
    got, problem = _call(sessions)
    if problem is None and got != 3:
        problem = "simultaneous ends and starts inflated the peak: process all ends before starts"
    if problem is not None:
        _fail(problem)


def test_staggered_chain_peaks_at_two() -> None:
    _verify([(i, i + 2) for i in range(6)])


def _must_raise_value_error(sessions: list[tuple[int, int]], mistake: str) -> None:
    try:
        with _time_box(SMALL_CALL_S):
            max_concurrent(sessions)
    except ValueError:
        return
    except _TooSlow:
        _fail(f"{mistake}; instead the call did not finish")
    except Exception as exc:
        _fail(f"{mistake}; instead it raised {type(exc).__name__}")
    _fail(mistake)


def test_zero_length_session_raises_value_error() -> None:
    _must_raise_value_error(
        [(1, 2), (5, 5)],
        "a session with start == end was accepted: a half-open session needs start < end; raise "
        "ValueError",
    )


def test_reversed_session_raises_value_error() -> None:
    _must_raise_value_error(
        [(3, 1)], "a session with start > end was accepted: raise ValueError when start >= end"
    )


def test_invalid_session_anywhere_in_the_list_raises_value_error() -> None:
    _must_raise_value_error(
        [(0, 4), (2, 6), (9, 9), (7, 8)],
        "an invalid session in the middle of the list was not reported: validate every session "
        "first",
    )


def test_sessions_list_and_tuples_are_not_mutated() -> None:
    sessions = [(6, 9), (1, 4), (3, 7)]
    before = list(sessions)
    _, problem = _call(sessions)
    if problem is not None:
        _fail(problem)
    if sessions != before:
        _fail("the input list was changed: sort a new list (sorted) instead of sorting in place")


def test_random_order_of_large_ties_gives_the_same_peak() -> None:
    base_sessions = [(t, t + 1) for t in range(40)] + [(t, t + 2) for t in range(40)]
    order = np.random.default_rng(3).permutation(len(base_sessions))
    _verify([base_sessions[int(i)] for i in order])


_CHILD = f"""
import random
from submission import max_concurrent
n = {LARGE_N}
sessions = [(i, i + 2) for i in range(n)]
random.Random(5).shuffle(sessions)
try:
    res = max_concurrent(sessions)
except Exception as exc:
    print("RAISED", type(exc).__name__)
else:
    print("RESULT", repr(res), type(res).__name__)
"""


def test_large_list_of_200000_sessions_finishes_within_the_time_limit() -> None:
    # Analytic answer: session i overlaps only sessions i - 1 and i + 1, and never all three at a
    # moment.
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
            f"too slow on {n} sessions: the function did not finish in {TIME_LIMIT_S:g} s. "
            f"Comparing each "
            "session with every other one is quadratic; sort the events once, then one pass"
        )
    lines = proc.stdout.strip().splitlines()
    line = lines[-1] if lines else ""
    if line.startswith("RAISED"):
        pytest.fail(f"your function raised on the large input: {line[len('RAISED ') :]}")
    if not line.startswith("RESULT"):
        pytest.fail("large run returned no result: check runtime errors and return type")
    if line != "RESULT 2 int":
        _fail(
            "the peak on the large input is wrong: the staggered sessions overlap at most in pairs"
        )
