"""Public tests: the examples from problem.md, the zero cases, and the negative-input errors."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from submission import longest_window_within_budget

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


def _result(costs: list[int], budget: int) -> Any:
    """The learner's answer. A hang or a crash fails the test with the mistake named."""
    if _HUNG:
        pytest.fail(_HUNG[0])
    try:
        with _time_box(SMALL_CALL_S):
            return longest_window_within_budget(costs, budget)
    except _TooSlow:
        message = (
            f"the call on {costs} with budget {budget} did not finish in {SMALL_CALL_S:g} s: "
            "each shrink step must subtract costs[left] and move left forward, so the loop ends"
        )
        _HUNG.append(message)
        pytest.fail(message)
    except Exception as exc:
        pytest.fail(
            f"your function raised {type(exc).__name__} on {costs} with budget {budget}: {exc}"
        )


def test_examples_from_problem_md() -> None:
    assert _result([1, 2, 3, 4, 5], 9) == (3, 0), (
        "[1, 2, 3, 4, 5] with budget 9: the longest valid windows have length 3, and the smallest "
        "start among them is 0"
    )
    assert _result([4, 1, 1, 1, 9], 3) == (3, 1), (
        "[4, 1, 1, 1, 9] with budget 3: [1, 1, 1] sums to exactly the budget, which is allowed"
    )
    assert _result([0, 0, 7, 0], 0) == (2, 0), (
        "zero-cost entries must count toward a window: [0, 0] is valid with budget 0"
    )
    assert _result([3, 1, 1, 1], 3) == (3, 1), (
        "after an entry that is too expensive, the window must restart on the right, not be lost: "
        "[1, 1, 1] sums to exactly 3"
    )


def test_no_valid_window_returns_zero_and_zero() -> None:
    assert _result([5, 6], 4) == (0, 0), (
        "every entry is above the budget, so there is no valid window: return (0, 0)"
    )
    assert _result([], 3) == (0, 0), "an empty list has no window: return (0, 0)"


def test_returns_a_tuple_of_two_python_ints() -> None:
    result = _result([2, 2, 2], 4)
    assert isinstance(result, tuple) and len(result) == 2, "return a (length, start) tuple"
    assert all(type(value) is int for value in result), (
        "return two plain Python ints, not NumPy scalars or floats"
    )


def test_negative_budget_raises_value_error() -> None:
    try:
        longest_window_within_budget([1, 2], -1)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a negative budget must raise ValueError, instead it raised {type(exc).__name__}"
        )
    pytest.fail("a negative budget was accepted: raise ValueError when budget < 0")
