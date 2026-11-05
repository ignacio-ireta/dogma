"""Public tests: the examples from problem.md, touching sessions, overlapping sessions, and the
empty input.

Each failure line names the mistake it guards against.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from submission import max_concurrent

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


def _result(sessions: list[tuple[int, int]]) -> Any:
    """The learner's answer. A hang or a crash fails the test with the mistake named."""
    if _HUNG:
        pytest.fail(_HUNG[0])
    try:
        with _time_box(SMALL_CALL_S):
            return max_concurrent(sessions)
    except _TooSlow:
        message = (
            f"the call on {len(sessions)} sessions did not finish in {SMALL_CALL_S:g} s: sort the "
            f"events "
            "once, then make one pass with a running count"
        )
        _HUNG.append(message)
        pytest.fail(message)
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on a small list: {exc}")


def test_overlapping_sessions_peak_at_their_common_moment() -> None:
    assert _result([(1, 4), (2, 6), (3, 5)]) == 3, (
        "(1, 4), (2, 6) and (3, 5) are all active at t = 3 and t = 4, so the peak is 3: count "
        "every "
        "session during its span, and keep the largest running count"
    )


def test_four_sessions_with_one_early_end_peak_at_three() -> None:
    assert _result([(0, 10), (2, 4), (3, 9), (8, 12)]) == 3, (
        "the peak of this example is 3: at t = 3 three sessions are active, and at t = 8 three "
        "are active, "
        "and no moment has four. A session that has ended (2, 4) must leave the count at t = 4"
    )


def test_no_sessions_has_peak_zero() -> None:
    assert _result([]) == 0, "no sessions means no active session: the peak is 0"


def test_touching_sessions_do_not_overlap() -> None:
    assert _result([(1, 3), (3, 5), (5, 7)]) == 1, (
        "touching sessions were counted as overlapping: a session that ends at t is no longer "
        "active at t, "
        "so at a shared moment apply the end before the start"
    )


def test_sessions_that_have_ended_are_not_counted() -> None:
    assert _result([(1, 2), (3, 4), (5, 6)]) == 1, (
        "sessions that have already ended are still counted: subtract a session when it ends"
    )


def test_peak_is_the_most_sessions_at_one_moment() -> None:
    assert _result([(0, 3), (1, 5), (2, 4), (4, 9)]) == 3, (
        "the peak was not found: track the running count after every event, and keep the largest "
        "value"
    )
