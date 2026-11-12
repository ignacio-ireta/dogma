"""Hidden tests: brute force on small random lists, an independent prefix-sum oracle on larger ones,
edge families, invalid input, no mutation, and a time limit on a 200,000-entry list.

Each failure message names the likely mistake first. A small call that runs too long is stopped
after SMALL_CALL_S seconds, so an endless loop fails with a named message instead of hanging the
run. The large-input test runs the learner's function in a child process killed at its limit.
"""

from __future__ import annotations

import ast
import bisect
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
from submission import longest_window_within_budget

from dogma_data import sample_window

LARGE_N = 200_000
TIME_LIMIT_S = 5.0
SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one small call runs past SMALL_CALL_S."""


HANG_MESSAGE = (
    "a call on a small list did not finish: check that each shrink step subtracts "
    "costs[left] and moves left forward, so the while loop ends"
)
# Once one call hangs, later calls fail at once with the same message, so a Hypothesis run
# does not spend the time limit on every example it draws while shrinking.
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


def _call(costs: list[int], budget: int) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(SMALL_CALL_S):
            return longest_window_within_budget(costs, budget), None
    except _TooSlow:
        _HUNG.append(HANG_MESSAGE)
        return None, HANG_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a small list: check the edge cases, "
            "an empty list and entries above the budget"
        )


def _brute(costs: list[int], budget: int) -> tuple[int, int]:
    """Independent answer: test every window. Quadratic, so only for small lists."""
    best_length, best_start = 0, 0
    n = len(costs)
    for start in range(n):
        total = 0
        for end in range(start, n):
            total += costs[end]
            if total <= budget and end - start + 1 > best_length:
                best_length, best_start = end - start + 1, start
    return best_length, best_start


def _prefix_oracle(costs: list[int], budget: int) -> tuple[int, int]:
    """Independent answer: prefix sums and binary search, O(n log n)."""
    prefix = [0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    best_length, best_start = 0, 0
    for right in range(1, len(prefix)):
        left = bisect.bisect_left(prefix, prefix[right] - budget)
        if right - left > best_length:
            best_length, best_start = right - left, left
    return best_length, best_start


def _first_problem(got: Any, want: tuple[int, int]) -> str | None:  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if not (isinstance(got, tuple) and len(got) == 2):
        return "the result must be a (length, start) tuple"
    if not all(type(value) is int for value in got):
        return "the result must hold two Python ints, not NumPy scalars or floats"
    if got != want and got == (want[1], want[0]):
        return "the fields are swapped: return (length, start), not (start, length)"
    if want[0] == 0 and got != (0, 0):
        return "no window is valid, so the result must be (0, 0)"
    if got[0] > want[0]:
        return (
            "the reported window is longer than any valid window: its sum must be at most the "
            "budget, so shrink from the left while the sum is above the budget (a while loop)"
        )
    if got[0] < want[0]:
        return (
            "the longest valid window is shorter than it should be: a window whose sum equals the "
            "budget is allowed, and a valid window must not be given up early"
        )
    if got[1] != want[1]:
        return (
            "same length, different start: among windows of the maximum length, return the "
            "smallest start (update the best only when the new window is strictly longer)"
        )
    return None


def _verify(costs: list[int], budget: int, want: tuple[int, int]) -> None:
    got, problem = _call(costs, budget)
    if problem is None:
        problem = _first_problem(got, want)
    if problem is not None:
        _fail(problem)


def _random_costs(rng: np.random.Generator, n: int, high: int) -> list[int]:
    """n real costs in a row: a run of a measured series, in whole units above its lowest value.

    A short list is monthly sea-surface temperature above 19 degrees (0 to 10); a long one is
    weekly CO2 above 313 ppm (0 to about 60), which climbs with a yearly wave. The bound is
    not used.
    """
    if n == 0:
        return []
    if n < 100:
        return [int(x) for x in np.round(sample_window("elnino", rng, n)) - 19]
    return [int(x) for x in np.round(sample_window("co2", rng, n)) - 313]


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_lists_match_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    costs = _random_costs(rng, int(rng.integers(0, 25)), high=10)
    budget = int(rng.integers(0, 40))
    _verify(costs, budget, _brute(costs, budget))


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_medium_random_lists_match_the_prefix_sum_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    costs = _random_costs(rng, int(rng.integers(100, 400)), high=60)
    budget = int(rng.integers(0, 3_000))
    _verify(costs, budget, _prefix_oracle(costs, budget))


def test_window_whose_sum_equals_the_budget_is_allowed() -> None:
    _verify([2, 2, 2], 4, (2, 0))


def test_equal_length_ties_return_the_smallest_start() -> None:
    _verify([1, 1, 1, 1], 2, (2, 0))


def test_no_valid_window_returns_zero_zero_for_all_too_expensive_entries() -> None:
    _verify([5, 6, 7], 4, (0, 0))


def test_empty_list_returns_zero_zero() -> None:
    _verify([], 0, (0, 0))


def test_all_zero_costs_give_the_whole_list_at_budget_zero() -> None:
    _verify([0, 0, 0, 0, 0], 0, (5, 0))


def test_long_run_at_the_end_beats_an_early_short_run() -> None:
    _verify([9, 9, 9, 1, 1, 1, 1], 4, (4, 3))


def _must_raise_value_error(costs: list[int], budget: int, mistake: str) -> None:
    try:
        with _time_box(SMALL_CALL_S):
            longest_window_within_budget(costs, budget)
    except ValueError:
        return
    except _TooSlow:
        _fail(f"{mistake}; instead the call did not finish")
    except Exception as exc:
        _fail(f"{mistake}; instead it raised {type(exc).__name__}")
    _fail(mistake)


def test_negative_cost_after_a_valid_prefix_raises_value_error() -> None:
    _must_raise_value_error(
        [1, 2, -1, 3], 10, "a negative cost was accepted: raise ValueError when any cost is below 0"
    )


def test_negative_budget_raises_value_error() -> None:
    _must_raise_value_error(
        [1], -1, "a negative budget was accepted: raise ValueError when budget < 0"
    )


def test_costs_list_is_not_mutated() -> None:
    costs = [4, 1, 1, 1, 9]
    before = list(costs)
    _, problem = _call(costs, 3)
    if problem is not None:
        _fail(problem)
    assert costs == before, "costs was changed in place: do not sort, pop or write into the input"


def test_one_large_entry_forces_several_removals() -> None:
    _verify([1, 1, 1, 1, 10], 3, (3, 0))


_CHILD = f"""
import sys
from submission import longest_window_within_budget
try:
    value = longest_window_within_budget([1] * {LARGE_N}, {LARGE_N // 2})
except Exception as exc:
    print("RAISED", type(exc).__name__, exc)
else:
    print("RESULT", repr(value))
"""


def test_large_list_of_200000_entries_finishes_within_the_time_limit() -> None:
    # The function runs in a child process that is killed at the limit. A worker thread could not
    # be stopped, and it would keep slowing the test session down after the failure.
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
            f"too slow on {LARGE_N} costs: the function did not finish in {TIME_LIMIT_S:g} s. "
            "Work that grows with n squared (checking every start, then extending each window) "
            "is the usual cause; move one left and one right pointer forward only"
        )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if line.startswith("RAISED"):
        pytest.fail(f"your function raised on the large input: {line[len('RAISED ') :]}")
    if not line.startswith("RESULT"):
        pytest.fail(f"the large run produced no result: {proc.stderr.strip()[-200:]}")
    try:
        value = ast.literal_eval(line[len("RESULT ") :])
    except (ValueError, SyntaxError):
        pytest.fail("the result could not be read back: return a (length, start) tuple of ints")
    problem = _first_problem(value, (LARGE_N // 2, 0))
    assert problem is None, problem


def test_single_entry_is_a_window_only_when_it_fits_the_budget() -> None:
    # The only window [5] sums to 5, which is above the budget 4, so no window is valid: (0, 0).
    _verify([5], 4, (0, 0))
    # The only window [3] sums to 3, which is within the budget 4: length 1, start 0.
    _verify([3], 4, (1, 0))
