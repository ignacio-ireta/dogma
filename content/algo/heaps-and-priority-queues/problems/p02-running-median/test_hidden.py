"""Hidden tests: a statistics.median oracle on prefixes, even and odd counts, orders and speed.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

from __future__ import annotations

import math
import signal
import statistics
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import running_median

LARGE_N = 60_000
LARGE_TIME_LIMIT_S = 6.0
MEDIAN_MISTAKE = (
    "a running median differs from the reference: the median of each prefix is the middle value "
    "for an odd count and the mean of the two middle values for an even count"
)


class _TooSlow(BaseException):
    """Raised inside the learner's call when it runs past its time box."""


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
    raise AssertionError(why)


def _expect_value_error(call: Callable[[], object], mistake: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        _fail(f"{mistake} (raised {type(exc).__name__} instead)")
    _fail(mistake)


def _solve(values: Sequence[float], *, limit: float = 5.0) -> list[float]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        with _time_box(limit):
            got = running_median(values)
    except _TooSlow:
        _fail("too slow on a long input: use two heaps, not a sort of every prefix")
    except Exception as exc:
        _fail(
            f"valid input raised {type(exc).__name__}: check the heap sizes, the sign of the "
            "max-heap and the even-count mean"
        )
    if not isinstance(got, list) or len(got) != len(values):
        _fail("return one median per prefix, as a list of the same length as the input")
    if not all(isinstance(m, float) for m in got):
        _fail("return every median as a Python float")
    return got


def _reference(values: Sequence[float]) -> list[float]:
    """Independent answer: statistics.median of every prefix."""
    return [float(statistics.median(values[: i + 1])) for i in range(len(values))]


def _matches(got: Sequence[float], want: Sequence[float]) -> bool:
    return len(got) == len(want) and all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("seed", range(30), ids=[f"ints-{s}" for s in range(30)])
def test_matches_statistics_median_on_random_integers(seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = [int(v) for v in rng.integers(-20, 21, size=int(rng.integers(1, 41)))]
    want = _reference(values)  # computed before the learner runs
    assert _matches(_solve(values), want), MEDIAN_MISTAKE


@pytest.mark.parametrize("seed", range(10), ids=[f"floats-{s}" for s in range(10)])
def test_matches_statistics_median_on_random_floats(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    values = [float(v) for v in rng.normal(scale=10.0, size=int(rng.integers(1, 41)))]
    want = _reference(values)
    assert _matches(_solve(values), want), MEDIAN_MISTAKE


def test_even_count_with_odd_sum_gives_a_half() -> None:
    assert _solve([1, 2]) == [1.0, 1.5], (
        "the even-count median must average the middle values without integer truncation"
    )
    assert _solve([3, 8]) == [3.0, 5.5], "the even-count median must preserve a fractional result"


def test_odd_count_gives_the_exact_middle_value() -> None:
    got = _solve([9, 1, 5, 7, 3])
    assert got == [9.0, 5.0, 5.0, 6.0, 5.0], (
        "the median must use the middle value for odd prefixes and average the "
        "middle pair for even prefixes"
    )


def test_ascending_and_descending_inputs() -> None:
    up = list(range(1, 31))
    assert _solve(up) == [(i + 2) / 2 for i in range(30)], (
        "the prefix 1..L has median (L + 1) / 2, so element i of the result is (i + 2) / 2"
    )
    assert _matches(_solve(up[::-1]), _reference(up[::-1])), MEDIAN_MISTAKE


def test_alternating_extremes() -> None:
    values = [0, 100, 1, 99, 2, 98, 3, 97]
    assert _matches(_solve(values), _reference(values)), (
        "alternating small and large values move the median between the two heaps"
    )


def test_large_magnitudes_with_small_differences() -> None:
    base = 1e12
    values = [base + offset for offset in (2.0, 0.0, 1.0, 3.0, 2.0, 0.5)]
    assert _matches(_solve(values), _reference(values)), (
        "large values with small differences must keep their order"
    )


def test_negative_and_mixed_numbers() -> None:
    values = [-3, 2.5, -1, 0, -7.25, 4, -2]
    assert _matches(_solve(values), _reference(values)), (
        "negative numbers and floats mixed with ints must be ordered as numbers"
    )


def test_duplicates_and_a_single_value() -> None:
    assert _solve([4]) == [4.0], "a single value is its own median"
    values = [5, 5, 1, 5, 5, 1]
    assert _matches(_solve(values), _reference(values)), "duplicates must count each time"


def test_empty_input_is_valid() -> None:
    assert _solve([]) == [], "an empty input has no prefixes, so the result is empty"


@pytest.mark.parametrize(
    "values",
    [[1.0, float("nan")], [float("inf"), 2.0], [2.0, float("-inf")]],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_non_finite_values_raise_value_error(values: list[float]) -> None:
    _expect_value_error(
        lambda: running_median(values),
        "a NaN or infinite value was accepted: raise ValueError, since it has no place in an order",
    )


def test_input_is_not_mutated() -> None:
    values = [3, 1, 4, 1, 5]
    before = list(values)
    _solve(values)
    assert values == before, (
        "the input list was reordered or changed: never sort or edit the caller's list"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.integers(-50, 50), min_size=1, max_size=25))
def test_each_median_lies_between_the_smallest_and_largest_value_so_far(values: list[int]) -> None:
    got = _solve(values)
    for i, median in enumerate(got):
        prefix = values[: i + 1]
        assert min(prefix) <= median <= max(prefix), (
            "a median must lie between the smallest and largest value of its prefix"
        )


def test_long_input_finishes_within_the_time_limit_and_matches_the_reference() -> None:
    rng = np.random.default_rng(8)
    values = [float(v) for v in rng.normal(size=LARGE_N)]
    checkpoints = [*range(999, LARGE_N, 6_000), LARGE_N - 1]
    want = {i: float(np.median(values[: i + 1])) for i in checkpoints}
    start = time.perf_counter()
    got = _solve(values, limit=LARGE_TIME_LIMIT_S)
    elapsed = time.perf_counter() - start
    assert elapsed < LARGE_TIME_LIMIT_S, (
        "a long input is too slow: each insertion must cost O(log n), not a sort of the prefix"
    )
    for i in checkpoints:
        assert math.isclose(got[i], want[i], rel_tol=1e-9, abs_tol=1e-9), MEDIAN_MISTAKE


def test_finite_extreme_even_median_does_not_overflow():
    got = _solve([1e308, 1e308, -1e308, -1e308])
    assert got == [1e308, 1e308, 1e308, 0.0], (
        "averaging finite middle values must not overflow: use a stable midpoint"
    )
