"""Hidden tests: a breadth-first search over sums as the oracle for small inputs, an independent
coin-by-coin table for larger ones, and the edge families named in problem.md.

Every failure names the kind of mistake and carries no hidden input or answer. A call that runs
past its time box fails with the timing message every time, so a slow learner gets the same named
failure on every replay.
"""

from __future__ import annotations

import random
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NoReturn

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import min_coins


class _TooSlow(BaseException):
    """Raised inside the learner's function when one call runs past its time box."""


SLOW_MESSAGE = (
    "a call did not finish in time: an unmemoised recursion tries every coin sequence (it is "
    "exponential), and a table rescanned on every step is quadratic; fill each entry once, "
    "in a loop, in increasing order of amount"
)
# Budgets are generous: the reference answers each call in well under a millisecond here.
CALL_BUDGET_S = 5.0
LARGE_AMOUNT = 50_000

# After one call times out, later calls fail at once with the same message, so a slow learner
# costs one budget instead of one budget per Hypothesis example.
_HUNG: list[bool] = []


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


def _call_within_budget(call: Callable[[], Any]) -> Any:
    """Run one learner call under the time box.

    A timeout leaves this function outside its except block, so the failure carries no chained
    exception. Hypothesis then sees the same failure on every replay of a slow example.
    """
    if _HUNG:
        _slow()
    try:
        with _time_box(CALL_BUDGET_S):
            return call()
    except _TooSlow:
        pass
    _HUNG.append(True)
    _slow()


def _answer(coins: Any, amount: Any) -> Any:
    try:
        return _call_within_budget(lambda: min_coins(coins, amount))
    except RecursionError:
        pytest.fail(
            f"min_coins raised RecursionError on amount {amount}: a memoised recursion goes "
            "as deep as the amount; build the table in a loop instead"
        )
    except Exception as exc:
        pytest.fail(f"min_coins raised {type(exc).__name__} on a valid input: {exc}")


def _bfs(coins: list[int], amount: int) -> int:
    """Independent oracle: breadth-first search over sums.

    Level k of the search holds the sums that need exactly k coins.
    """
    if amount == 0:
        return 0
    seen = {0}
    frontier: deque[tuple[int, int]] = deque([(0, 0)])
    while frontier:
        total, used = frontier.popleft()
        for coin in coins:
            nxt = total + coin
            if nxt == amount:
                return used + 1
            if nxt < amount and nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, used + 1))
    return -1


def _coin_by_coin(coins: list[int], amount: int) -> int:
    """Independent oracle: a different formulation. Process one coin type at a time, unbounded."""
    inf = float("inf")
    best: list[float] = [0.0] + [inf] * amount
    for coin in sorted(set(coins)):
        for v in range(coin, amount + 1):
            best[v] = min(best[v], best[v - coin] + 1)
    return -1 if best[amount] == inf else int(best[amount])


MISTAKE = (
    "min_coins differs from the reference: check the base case, the minimum over every coin, and "
    "the order in which the table is filled"
)


@settings(max_examples=120, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_inputs_match_breadth_first_search(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic inputs, not security.
    coins = [rng.randrange(1, 10) for _ in range(rng.randrange(0, 5))]
    amount = rng.randrange(0, 41)
    got = _answer(coins, amount)
    _check(got == _bfs(coins, amount), MISTAKE)


def test_random_larger_inputs_match_the_coin_by_coin_table() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic inputs, not security.
    for _ in range(40):
        coins = [rng.randrange(1, 60) for _ in range(rng.randrange(1, 8))]
        amount = rng.randrange(0, 600)
        _check(_answer(coins, amount) == _coin_by_coin(coins, amount), MISTAKE)


def test_zero_amount_is_zero_for_any_denominations() -> None:
    for coins in ([], [1], [7, 11], [3, 3, 3]):
        _check(
            _answer(coins, 0) == 0,
            "an amount of 0 must return 0 for every denomination set",
        )


def test_greedy_trap_returns_the_true_minimum() -> None:
    _check(
        _answer([1, 3, 4], 6) == 2,
        "the fewest coins for 6 with 1, 3 and 4 is two (3 + 3), not three",
    )


def test_unreachable_amounts_return_minus_one_not_zero_or_a_count() -> None:
    _check(
        _answer([4, 6], 7) == -1,
        "an amount that no combination reaches must return -1",
    )
    _check(
        _answer([], 5) == -1,
        "with no coins, any positive amount is impossible and must return -1",
    )


def test_repeated_and_unsorted_coins_do_not_change_the_answer() -> None:
    expected = _answer([1, 5, 10], 27)
    _check(
        _answer([10, 5, 1, 5, 10, 1], 27) == expected,
        "repeated or unsorted coin values must not change the minimum count",
    )


def test_coin_order_is_irrelevant_for_every_permutation_tried() -> None:
    coins = [2, 3, 7]
    expected = _answer(coins, 23)
    for ordering in ([7, 3, 2], [3, 2, 7], [2, 7, 3]):
        _check(
            _answer(ordering, 23) == expected,
            "the order of coins must not change the minimum count",
        )


def test_large_amount_finishes_within_the_time_limit() -> None:
    expected = _coin_by_coin([1, 7, 13], LARGE_AMOUNT)
    _check(_answer([1, 7, 13], LARGE_AMOUNT) == expected, MISTAKE)


def test_large_amount_with_large_denominations_is_exact() -> None:
    coins = [997, 1009, 15]
    expected = _coin_by_coin(coins, 40_000)
    _check(_answer(coins, 40_000) == expected, MISTAKE)


def test_return_type_is_a_python_int() -> None:
    result = _answer([2, 5], 9)
    _check(
        type(result) is int,
        "return a Python int: the count must not be a float, a bool or a NumPy scalar",
    )


@pytest.mark.parametrize(
    ("coins", "amount"),
    [
        ([1, 2.0], 3),
        ([1, "2"], 3),
        ([True, 2], 3),
        ([1, None], 3),
        ([1, 2], 2.0),
        ([1, 2], "3"),
        ("12", 3),
    ],
    ids=[
        "float-coin",
        "string-coin",
        "bool-coin",
        "none-coin",
        "float-amount",
        "string-amount",
        "coins-not-a-list",
    ],
)
def test_non_int_input_raises_type_error(coins: Any, amount: Any) -> None:
    try:
        _call_within_budget(lambda: min_coins(coins, amount))
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"a non-int input must raise TypeError, instead it raised {type(exc).__name__}")
    pytest.fail("a non-int input was accepted: validate coins and amount with type(...) is int")


@pytest.mark.parametrize(
    ("coins", "amount"),
    [([0, 2], 4), ([-1, 2], 4), ([1, 2], -1)],
    ids=["zero-coin", "negative-coin", "negative-amount"],
)
def test_invalid_values_raise_value_error(coins: list[int], amount: int) -> None:
    try:
        _call_within_budget(lambda: min_coins(coins, amount))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"an invalid value must raise ValueError, instead it raised {type(exc).__name__}"
        )
    pytest.fail(
        "an invalid value was accepted: a coin must be at least 1 and the amount at least 0"
    )


def test_coins_list_is_not_mutated() -> None:
    coins = [5, 2, 3]
    before = list(coins)
    _answer(coins, 11)
    _check(
        coins == before,
        "the coins list was changed: sort or copy it instead of editing the input",
    )


def test_coins_given_as_a_tuple_raises_type_error() -> None:
    try:
        _call_within_budget(lambda: min_coins((1, 2), 3))  # type: ignore[arg-type]
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(
            f"coins that is not a list must raise TypeError, instead it raised {type(exc).__name__}"
        )
    pytest.fail(
        "coins that is not a list was accepted: check that coins is a list before reading it"
    )


def test_bool_amount_raises_type_error() -> None:
    try:
        _call_within_budget(lambda: min_coins([1, 2], True))  # type: ignore[arg-type]
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"a bool amount must raise TypeError, instead it raised {type(exc).__name__}")
    pytest.fail("a bool amount was accepted: amount must be a Python int, and bool is not one")
