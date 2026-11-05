"""Public tests: the examples from problem.md, a window of width one, and an invalid width.

Each failure line names the mistake it guards against.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from submission import sliding_window_max

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


def _result(values: list[float], k: int) -> Any:
    """The learner's answer. A hang or a crash fails the test with the mistake named."""
    if _HUNG:
        pytest.fail(_HUNG[0])
    try:
        with _time_box(SMALL_CALL_S):
            return sliding_window_max(values, k)
    except _TooSlow:
        message = (
            f"the call on {len(values)} values with k = {k} did not finish in {SMALL_CALL_S:g} s: "
            "each index must enter and leave the deque once, so the loop ends"
        )
        _HUNG.append(message)
        pytest.fail(message)
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on a small list: {exc}")


def _explain(got: Any, want: list[float], values: list[float], k: int) -> str | None:
    """Name the mistake behind a wrong answer on a public example, or return None."""
    if not isinstance(got, list) or len(got) != len(want):
        return (
            "the result must have len(values) - k + 1 entries, one per window, starting with the "
            "first "
            "window: a window is missing or extra"
        )
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
            "a window misses its largest value: a value is dropped while it is still a window "
            "maximum"
        )
    return None


def test_examples_from_problem_md() -> None:
    examples = [
        ([4.0, 2.0, 12.0, 11.0, -5.0], 2, [4.0, 12.0, 12.0, 11.0]),
        ([7.0, 7.0, 1.0, 0.5], 2, [7.0, 7.0, 1.0]),
        ([3.0, -2.0, 8.0], 3, [8.0]),
    ]
    for values, k, want in examples:
        problem = _explain(_result(values, k), want, values, k)
        assert problem is None, problem


def test_window_of_width_one_returns_the_values_unchanged() -> None:
    values = [2.5, -1.0, 0.0, 9.0]
    problem = _explain(_result(list(values), 1), values, values, 1)
    assert problem is None, problem


def test_width_larger_than_the_list_raises_value_error() -> None:
    try:
        sliding_window_max([1.0, 2.0], 3)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a width larger than the list must raise ValueError, instead it raised "
            f"{type(exc).__name__}"
        )
    pytest.fail(
        "a width larger than the list was accepted: raise ValueError when k > len(values), "
        "instead of returning an empty list"
    )
