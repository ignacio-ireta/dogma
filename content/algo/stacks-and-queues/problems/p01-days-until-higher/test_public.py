"""Public tests: the examples from problem.md, equal readings, negative readings, empty input
and types."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from submission import days_until_higher

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


def _result(values: list[float]) -> Any:
    """The learner's answer. A hang or a crash fails the test with the mistake named."""
    if _HUNG:
        pytest.fail(_HUNG[0])
    try:
        with _time_box(SMALL_CALL_S):
            return days_until_higher(values)
    except _TooSlow:
        message = (
            f"the call on a list of {len(values)} readings did not finish in {SMALL_CALL_S:g} s: "
            "move the index forward once and pop each waiting index at most once"
        )
        _HUNG.append(message)
        pytest.fail(message)
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on a small list: {exc}")


def _explain(got: Any, want: list[int], values: list[float]) -> str | None:  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    """Name the mistake behind a wrong answer on a public example, or return None."""
    if not isinstance(got, list) or len(got) != len(want):
        return "the result must be a list with one entry per reading, the same length as the input"
    if got == want[::-1] and want != want[::-1]:
        return (
            "the entries are in reverse order: a right-to-left scan must fill entry i, or reverse "
            "the result"
        )
    for i, (entry, expected) in enumerate(zip(got, want, strict=True)):
        if entry == expected:
            continue
        if expected == 0:
            return (
                "a reading with no strictly larger reading after it must get 0, not -1 or another "
                "value"
            )
        if entry == i + expected:
            return (
                "an entry holds the position of the higher reading, not the number of steps to it: "
                "subtract the current index"
            )
        if entry < expected:
            ender = i + entry
            if entry != 0 and values[ender] == values[i]:
                return (
                    "a later reading equal to this one ended its wait: use > and not >= in the "
                    "pop test"
                )
            return (
                "a reading was resolved too early or never: pop every waiting reading that is "
                "strictly "
                "lower than the new one (a while loop, not an if)"
            )
        return "a reading waited too long: check the strict comparison and the order of the pops"
    return None


def test_examples_from_problem_md() -> None:
    examples = [
        ([5.0, 3.0, 8.0, 8.0, 2.0, 9.0], [2, 1, 3, 2, 1, 0]),
        ([1.0, 1.0, 1.0], [0, 0, 0]),
        ([-2.5, -1.0, -3.0], [1, 0, 0]),
        ([], []),
    ]
    for values, want in examples:
        problem = _explain(_result(values), want, values)
        assert problem is None, problem


def test_equal_readings_do_not_count_as_higher() -> None:
    assert _result([2.0, 2.0, 2.0, 3.0]) == [3, 2, 1, 0], (
        "a later reading equal to this one was counted as higher: only a strictly larger reading "
        "ends the wait, so compare with > and not >="
    )


def test_returns_a_list_of_python_ints() -> None:
    result = _result([4.0, 1.0, 6.0])
    assert isinstance(result, list), "return a list of ints, one entry per reading"
    assert all(type(entry) is int for entry in result), (
        "each entry must be a plain Python int, not a float or a NumPy scalar"
    )
