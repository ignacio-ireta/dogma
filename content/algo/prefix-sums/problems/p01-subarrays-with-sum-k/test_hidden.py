"""Hidden tests: brute force on small lists and a position-based oracle on larger ones.

Edge families cover exact integers, validation, mutation and a large-input time limit.

Every failure message names the mistake without carrying a hidden input or answer.
A call that runs past its time box is stopped by a signal, so slow loops fail by name.
The previous signal handler is restored after each call.
"""

from __future__ import annotations

import bisect
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import count_subarrays_with_sum

LARGE_N = 200_000
SMALL_CALL_S = 2.0
LARGE_CALL_S = 4.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when a call runs past its time box."""


SLOW_MESSAGE = (
    "a call did not finish in time: checking every pair of positions is the usual cause; one "
    "pass that keeps a dictionary of earlier prefix sums runs in linear time"
)
# Once one call hangs, later calls fail at once with the same message, so a Hypothesis run does not
# spend the time limit on every example it draws while it shrinks.
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


def _call(nums: list[int], k: Any, limit: float = SMALL_CALL_S) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(limit):
            return count_subarrays_with_sum(nums, k), None
    except _TooSlow:
        _HUNG.append(SLOW_MESSAGE)
        return None, SLOW_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a valid input: check the running total, "
            "the dictionary lookup and the loop bounds"
        )


def _brute(nums: list[int], k: int) -> int:
    """Independent answer: sum every pair of positions. Quadratic, so only for small lists."""
    total = 0
    for i in range(len(nums)):
        running = 0
        for j in range(i, len(nums)):
            running += nums[j]
            if running == k:
                total += 1
    return total


def _positions_oracle(nums: list[int], k: int) -> int:
    """Independent answer: group prefix sums by value, then binary-search the earlier positions.

    It uses no running dictionary: for each j it counts the positions i < j whose prefix sum equals
    P_j - k, with bisect over the sorted positions of that value.
    """
    prefix = [0]
    for value in nums:
        prefix.append(prefix[-1] + value)
    where: dict[int, list[int]] = {}
    for index, value in enumerate(prefix):
        where.setdefault(value, []).append(index)
    total = 0
    for j, value in enumerate(prefix):
        positions = where.get(value - k, [])
        total += bisect.bisect_left(positions, j)
    return total


def _first_problem(got: Any, expected: int) -> str | None:
    """Return the first mistake found, or None. One return path keeps the reports stable."""
    if type(got) is not int:
        return "return a Python int: the count must not be a float or a NumPy scalar"
    if got > expected:
        return (
            "too many subarrays: the sign in P_j - k, or a lookup that runs after P_j is inserted, "
            "counts subarrays that do not sum to k"
        )
    if got < expected:
        return (
            "too few subarrays: the empty prefix 0 must be counted once before the loop, and every "
            "prefix must be recorded in the dictionary"
        )
    return None


def _verify(nums: list[int], k: int, expected: int, limit: float = SMALL_CALL_S) -> None:
    got, problem = _call(nums, k, limit)
    if problem is None:
        problem = _first_problem(got, expected)
    if problem is not None:
        _fail(problem)


def _random_list(rng: np.random.Generator, n: int, low: int, high: int) -> list[int]:
    return [int(v) for v in rng.integers(low, high + 1, size=n)]


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_lists_match_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    nums = _random_list(rng, int(rng.integers(0, 12)), -4, 4)
    k = int(rng.integers(-6, 7))
    _verify(nums, k, _brute(nums, k))


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_medium_random_lists_match_the_position_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    nums = _random_list(rng, int(rng.integers(100, 400)), -20, 20)
    k = int(rng.integers(-40, 41))
    _verify(nums, k, _positions_oracle(nums, k))


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_count_stays_between_zero_and_all_subarrays(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 30))
    nums = _random_list(rng, n, -3, 3)
    got, problem = _call(nums, int(rng.integers(-3, 4)))
    if problem is not None:
        _fail(problem)
    assert type(got) is int, "return a Python int: the count must not be a float or a NumPy scalar"
    assert 0 <= got <= n * (n + 1) // 2, (
        "the count is outside 0 .. n(n+1)/2: each subarray must be counted at most once"
    )


def test_empty_list_has_no_subarrays_for_any_target() -> None:
    _verify([], 0, 0)
    _verify([], -7, 0)


def test_all_zero_entries_with_zero_target_count_every_subarray() -> None:
    n = 50
    _verify([0] * n, 0, n * (n + 1) // 2)


def test_all_zero_entries_never_match_a_nonzero_target() -> None:
    _verify([0] * 40, 1, 0)


def test_subarray_starting_at_the_first_position_is_counted() -> None:
    _verify([3, 1], 3, 1)
    _verify([5], 5, 1)


def test_no_subarray_reaches_the_target() -> None:
    _verify([1, 2, 3], 100, 0)


def test_negative_target_with_all_negative_entries() -> None:
    _verify([-1, -2, -3, -1], -3, _brute([-1, -2, -3, -1], -3))


def test_mixed_signs_and_repeated_prefix_values() -> None:
    nums = [2, -2, 2, -2, 2]
    _verify(nums, 0, _brute(nums, 0))
    _verify(nums, 2, _brute(nums, 2))


def test_very_large_values_keep_exact_integer_counts() -> None:
    nums = [10**16, 1, 1, -(10**16), 1, 10**18, -(10**18), 1]
    for k in (1, 2, 10**16 + 1):
        _verify(nums, k, _brute(nums, k))


def test_inputs_are_not_mutated() -> None:
    nums = [4, -1, 2, 1, -3]
    before = list(nums)
    _, problem = _call(nums, 3)
    if problem is not None:
        _fail(problem)
    assert nums == before, "nums was changed in place: do not insert, sort or pop from the input"


@pytest.mark.parametrize(
    ("nums", "k"),
    [
        ([1, 2.0, 3], 3),
        ([1, "2", 3], 3),
        ([1, None, 3], 3),
        ([1, True], 1),
        ([np.int64(1), 2], 3),
        ([1, 2], 2.0),
        ([1, 2], "2"),
    ],
    ids=[
        "float-entry",
        "string-entry",
        "none-entry",
        "bool-entry",
        "numpy-int-entry",
        "float-target",
        "string-target",
    ],
)
def test_non_int_input_raises_type_error(nums: list[Any], k: Any) -> None:
    try:
        count_subarrays_with_sum(nums, k)
    except TypeError:
        return
    except Exception as exc:
        _fail(f"a non-int input must raise TypeError, instead it raised {type(exc).__name__}")
    _fail("a non-int input was accepted: check k and every entry with type(value) is int")


def test_large_list_finishes_within_the_time_limit() -> None:
    rng = np.random.default_rng(11)
    nums = _random_list(rng, LARGE_N, -3, 3)
    expected = _positions_oracle(nums, 0)
    got, problem = _call(nums, 0, LARGE_CALL_S)
    if problem is not None:
        _fail(
            "too slow on a long list: a nested loop over pairs of positions is the usual cause; "
            "keep one pass with a dictionary"
            if problem == SLOW_MESSAGE
            else problem
        )
    if type(got) is not int or got != expected:
        _fail(_first_problem(got, expected) or "wrong count on a long list")
