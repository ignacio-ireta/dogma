"""Public tests: the examples from problem.md, zero amount, and one invalid input.

A call that runs past its time box fails by name, instead of hanging the run.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NoReturn

import pytest
from submission import min_coins


class _TooSlow(BaseException):
    """Raised inside the learner's function when one call runs past its time box."""


SLOW_MESSAGE = (
    "a call did not finish in time: an unmemoised recursion tries every coin sequence (it is "
    "exponential), and a table rescanned on every step is quadratic; fill each entry once, "
    "in a loop, in increasing order of amount"
)
PUBLIC_BUDGET_S = 5.0  # the reference takes well under a millisecond on these inputs


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


def _slow() -> NoReturn:
    pytest.fail(SLOW_MESSAGE)


def _check(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _answer(coins: Any, amount: Any) -> Any:
    try:
        with _time_box(PUBLIC_BUDGET_S):
            return min_coins(coins, amount)
    except _TooSlow:
        pass  # failed below, outside this except block, so no chained exception is reported
    except RecursionError:
        pytest.fail(
            f"min_coins raised RecursionError on amount {amount}: a memoised recursion goes "
            "as deep as the amount; build the table in a loop instead"
        )
    except Exception as exc:
        pytest.fail(f"min_coins raised {type(exc).__name__} on a valid input: {exc}")
    _slow()


def test_three_coins_for_eleven_with_a_five_and_a_one() -> None:
    _check(
        _answer([1, 2, 5], 11) == 3,
        "[1, 2, 5] for 11: 5 + 5 + 1 uses three coins; the answer must be the fewest, "
        "not the count of some other combination",
    )


def test_greedy_largest_first_is_not_always_fewest() -> None:
    _check(
        _answer([1, 3, 4], 6) == 2,
        "[1, 3, 4] for 6: 3 + 3 uses two coins. Taking the largest coin first (4 + 1 + 1) uses "
        "three, so the answer must consider every coin at each value",
    )


def test_impossible_amount_returns_minus_one() -> None:
    _check(
        _answer([2], 3) == -1,
        "[2] for 3: no combination of even coins makes an odd amount, so the answer is -1",
    )


def test_zero_amount_needs_zero_coins_even_without_coins() -> None:
    _check(
        _answer([], 0) == 0,
        "an amount of 0 needs 0 coins, even when there are no coin types",
    )
    _check(
        _answer([5, 10], 0) == 0,
        "an amount of 0 needs 0 coins for any denominations",
    )
