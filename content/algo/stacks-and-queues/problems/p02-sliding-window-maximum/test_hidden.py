"""Hidden tests: brute force on random lists, expiry families, invalid widths, no mutation, and
a time limit.

The oracle takes the maximum of every window from scratch, which is quadratic, so it runs only on
small
and medium lists. The 200,000-value test runs the learner's function in a child process killed at
its limit.
Each failure line names the likely mistake first.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import sliding_window_max

LARGE_N = 200_000
LARGE_K = 50_000
TIME_LIMIT_S = 5.0
SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one small call runs past SMALL_CALL_S."""


HANG_MESSAGE = (
    "a call on a small list did not finish: check that the deque loop pops from each end at most "
    "once per "
    "index, and that the loop over the values ends"
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


def _call(values: list[float], k: int) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(SMALL_CALL_S):
            return sliding_window_max(values, k), None
    except _TooSlow:
        _HUNG.append(HANG_MESSAGE)
        return None, HANG_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a small list: check the window width "
            f"and "
            "the window bounds"
        )


def _brute(values: list[float], k: int) -> list[float]:
    """Independent answer: the maximum of every window, computed from scratch."""
    return [max(values[i : i + k]) for i in range(len(values) - k + 1)]


def _first_problem(  # noqa: PLR0911 - one return per learner mistake
    got: Any, want: list[float], values: list[float], k: int
) -> str | None:
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if not isinstance(got, list):
        return "the result must be a list with one entry per window"
    if len(got) != len(want):
        return "the result must have len(values) - k + 1 entries, one per window"
    if not all(type(entry) is float for entry in got):
        return "each maximum must be an input float unchanged, not a rounded or cast value"
    for i, (entry, expected) in enumerate(zip(got, want, strict=True)):
        if entry == expected:
            continue
        if entry > expected:
            return (
                "a window reports a value that has already left it: drop the front index of the "
                "deque "
                "once it falls outside the window"
            )
        if entry == values[i + k - 1]:
            return (
                "a window reports its last value instead of its largest: drop the smaller values "
                "from the "
                "back before adding the new one"
            )
        return (
            "a window misses its largest value: a value is dropped from the deque while it is "
            "still the "
            "maximum of a window"
        )
    return None


def _verify(values: list[float], k: int) -> None:
    want = _brute(
        values, k
    )  # computed before the learner runs, so an edit to values cannot skew it
    got, problem = _call(values, k)
    if problem is None:
        problem = _first_problem(got, want, values, k)
    if problem is not None:
        _fail(problem)


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_lists_match_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 16))
    values = [float(x) for x in rng.integers(-5, 8, size=n)]  # small range: ties are common
    k = int(rng.integers(1, n + 1))
    _verify(values, k)


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_medium_random_lists_match_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(60, 300))
    values = [float(x) for x in rng.integers(-1000, 1000, size=n)]
    k = int(rng.integers(1, 60))
    _verify(values, k)


def test_window_as_wide_as_the_list_gives_one_entry() -> None:
    _verify([3.0, -2.0, 8.0, 8.0, 1.0], 5)


def test_window_of_width_one_returns_the_input() -> None:
    values = [0.5, -4.0, 2.0, 2.0]
    got, problem = _call(list(values), 1)
    if problem is None and got != values:
        problem = "a window of width 1 must return the input values: each window holds one value"
    if problem is not None:
        _fail(problem)


def test_a_maximum_that_leaves_the_window_is_replaced_by_a_smaller_value() -> None:
    values = [9.0, 1.0, 2.0, 3.0]
    got, problem = _call(values, 2)
    if problem is None and got != [9.0, 2.0, 3.0]:
        problem = (
            "a value that has left the window is still reported: drop the front index when it is "
            "outside "
            "the window"
        )
    if problem is not None:
        _fail(problem)


def test_a_maximum_stays_for_exactly_k_positions_then_leaves() -> None:
    values = [5.0, 1.0, 1.0, 0.5]
    got, problem = _call(values, 2)
    if problem is None and got != [5.0, 1.0, 1.0]:
        problem = (
            "the window is the wrong width: a maximum belongs to the window only for k positions, "
            "so the "
            "index i - k must be dropped when the window moves to i"
        )
    if problem is not None:
        _fail(problem)


def test_equal_maxima_across_a_slide_are_all_reported() -> None:
    values = [4.0, 4.0, 1.0, 1.0]
    got, problem = _call(values, 2)
    if problem is None and got != [4.0, 4.0, 1.0]:
        problem = (
            "equal maxima across a slide are wrong: a value equal to the window maximum "
            "is still a maximum"
        )
    if problem is not None:
        _fail(problem)


def _must_raise_value_error(values: list[float], k: int, mistake: str) -> None:
    try:
        with _time_box(SMALL_CALL_S):
            sliding_window_max(values, k)
    except ValueError:
        return
    except _TooSlow:
        _fail(f"{mistake}; instead the call did not finish")
    except Exception as exc:
        _fail(f"{mistake}; instead it raised {type(exc).__name__}")
    _fail(mistake)


def test_zero_width_raises_value_error() -> None:
    _must_raise_value_error([1.0, 2.0], 0, "a width of 0 was accepted: raise ValueError when k < 1")


def test_negative_width_raises_value_error() -> None:
    _must_raise_value_error(
        [1.0, 2.0], -2, "a negative width was accepted: raise ValueError when k < 1"
    )


def test_width_larger_than_the_list_raises_value_error() -> None:
    _must_raise_value_error(
        [1.0, 2.0, 3.0],
        4,
        "a width larger than the list was accepted: raise ValueError when k > len(values)",
    )


def test_empty_list_raises_value_error() -> None:
    _must_raise_value_error(
        [], 1, "an empty list was accepted: every width is larger than 0, so raise ValueError"
    )


def test_values_list_is_not_mutated() -> None:
    values = [4.0, 2.0, 12.0, 11.0, -5.0]
    before = list(values)
    _, problem = _call(values, 2)
    if problem is not None:
        _fail(problem)
    if values != before:
        _fail(
            "the input list was changed: do not pop from or write into values; keep the indices "
            "in the deque"
        )


_CHILD = f"""
from submission import sliding_window_max
n = {LARGE_N}
k = {LARGE_K}
values = [float(i) for i in range(n)]
try:
    res = sliding_window_max(values, k)
except Exception as exc:
    print("RAISED", type(exc).__name__)
else:
    print("RESULT", len(res), int(res[0]), int(res[-1]), int(sum(res)))
"""


def test_large_list_of_200000_values_finishes_within_the_time_limit() -> None:
    # Analytic answer: the values rise by one each step, so each window's maximum is its last value.
    n, k = LARGE_N, LARGE_K
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
            f"too slow on {n} values with k = {k}: the function did not finish in "
            f"{TIME_LIMIT_S:g} s. Taking "
            "the maximum of every window from scratch is O(n * k); a deque keeps the work O(n)"
        )
    lines = proc.stdout.strip().splitlines()
    line = lines[-1] if lines else ""
    if line.startswith("RAISED"):
        pytest.fail(f"your function raised on the large input: {line[len('RAISED ') :]}")
    fields = line.split()
    if not fields or fields[0] != "RESULT":
        pytest.fail("large run returned no result: check runtime errors and return type")
    length, first, last, total = (int(x) for x in fields[1:])
    if length != n - k + 1:
        _fail("the result must have len(values) - k + 1 entries, one per window")
    if first != k - 1 or last != n - 1:
        _fail(
            "a window on the large input reports the wrong maximum: each window ends at its "
            "largest value"
        )
    if total != sum(range(k - 1, n)):
        _fail(
            "the window maxima on the large input do not add up: check the window bounds and the "
            "front"
        )


def test_full_width_window_does_not_reorder_the_input() -> None:
    values = [4.0, 2.0, 12.0, 11.0, -5.0]
    before = list(values)
    _, problem = _call(values, len(values))
    if problem is not None:
        _fail(problem)
    if values != before:
        _fail(
            "the input list was changed when the window is as wide as the list: do not sort or "
            "reorder values, even in that case"
        )
