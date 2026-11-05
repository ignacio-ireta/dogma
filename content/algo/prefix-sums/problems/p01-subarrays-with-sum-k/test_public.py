"""Public tests: the examples from problem.md, the empty list, and the type rule.

A slow call is stopped after SMALL_CALL_S seconds with a named message.
The previous signal handler is restored afterwards.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from submission import count_subarrays_with_sum

SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one call runs past SMALL_CALL_S."""


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


def _count(nums: list[int], k: Any) -> Any:
    """The learner's answer. A hang or a crash fails the test with the mistake named."""
    if _HUNG:
        pytest.fail(_HUNG[0])
    try:
        with _time_box(SMALL_CALL_S):
            return count_subarrays_with_sum(nums, k)
    except _TooSlow:
        message = (
            f"the call on {nums} with k = {k} did not finish in {SMALL_CALL_S:g} s: checking every "
            "pair of positions is too slow; one pass with a dictionary of earlier "
            "prefix sums is linear"
        )
        _HUNG.append(message)
        pytest.fail(message)
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on {nums} with k = {k}: {exc}")


def test_examples_from_problem_md() -> None:
    assert _count([1, 1, 1], 2) == 2, (
        "[1, 1, 1] with k = 2: the subarrays [1, 1] at positions 0-1 and 1-2 both count"
    )
    assert _count([1, -1, 1, -1], 0) == 4, (
        "[1, -1, 1, -1] with k = 0: [1, -1], [-1, 1], [1, -1] and the whole list all count; a "
        "lookup that matches the current prefix with itself would count more"
    )
    assert _count([-2, 3, -1], -1) == 1, (
        "[-2, 3, -1] with k = -1: only [-1] at position 2 sums to -1; negative values must not "
        "break the count"
    )


def test_empty_list_has_no_subarrays() -> None:
    assert _count([], 5) == 0, "an empty list has no subarrays, so the count is 0"


def test_float_target_raises_type_error() -> None:
    try:
        count_subarrays_with_sum([1, 2], 1.0)  # type: ignore[arg-type]  # invalid target
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"a float target must raise TypeError, instead it raised {type(exc).__name__}")
    pytest.fail("a float target was accepted: check that k is a Python int and raise TypeError")
