"""Hidden tests: brute force on small and medium random lists, edge families, no mutation, and a
time limit.

Each failure line names the likely mistake first. A small call that runs too long is stopped after
SMALL_CALL_S seconds, so an endless loop fails with a named message. The 200,000-reading test runs
the
learner's function in a child process that is killed at its limit.
"""

from __future__ import annotations

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
from submission import days_until_higher

from dogma_data import rng as data_rng
from dogma_data import sample_window

LARGE_N = 200_000
TIME_LIMIT_S = 5.0
SMALL_CALL_S = 2.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when one small call runs past SMALL_CALL_S."""


HANG_MESSAGE = (
    "a call on a small list did not finish: check that the index moves forward once per reading "
    "and "
    "that every pop removes one waiting index"
)
# Once one call hangs, later calls fail at once with the same message, so a Hypothesis run does not
# spend the time limit on every example it draws while shrinking.
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


def _call(values: list[float]) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(SMALL_CALL_S):
            return days_until_higher(values), None
    except _TooSlow:
        _HUNG.append(HANG_MESSAGE)
        return None, HANG_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on a small list: check the empty list, "
            "a single reading, and negative values"
        )


def _brute(values: list[float]) -> list[int]:
    """Independent answer: for each index, scan forward for the first strictly larger reading."""
    out: list[int] = []
    n = len(values)
    for i in range(n):
        answer = 0
        for j in range(i + 1, n):
            if values[j] > values[i]:
                answer = j - i
                break
        out.append(answer)
    return out


def _first_problem(got: Any, want: list[int], values: list[float]) -> str | None:  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if not isinstance(got, list):
        return "the result must be a list with one entry per reading"
    if len(got) != len(want):
        return "the result must have one entry per reading: the same length as the input list"
    if not all(type(entry) is int for entry in got):
        return "each entry must be a plain Python int, not a float or a NumPy scalar"
    if got == want[::-1] and want != want[::-1]:
        return (
            "the entries are in reverse order: a right-to-left scan must fill entry i, or reverse "
            "the result"
        )
    if any(entry < 0 or entry >= len(values) - i for i, entry in enumerate(got)):
        return (
            "a wait distance is outside the available future positions: use a nonnegative distance"
        )
    for i, (entry, expected) in enumerate(zip(got, want, strict=True)):
        if entry == expected:
            continue
        if expected == 0:
            return (
                "a reading with no strictly larger reading after it must get 0 (not -1 or another "
                "value)"
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
                "lower than the new one, and only a strictly larger reading ends a wait"
            )
        return (
            "a reading waited too long: the first strictly larger later reading ends its wait, so "
            "check the strict comparison and the order of the pops"
        )
    return None


def _verify(values: list[float]) -> None:
    want = _brute(
        values
    )  # computed before the learner runs, so a learner that edits its input cannot skew it
    got, problem = _call(values)
    if problem is None:
        problem = _first_problem(got, want, values)
    if problem is not None:
        _fail(problem)


def _random_readings(rng: np.random.Generator, n: int, low: int, high: int) -> list[float]:
    """n real monthly sea-surface temperatures in a row, in whole degrees above or below 24.

    Whole degrees, so equal readings (ties) are common, and the season makes runs that rise and
    fall. Readings below 24 degrees come out negative, on purpose. The bounds are not used: the
    range is the one the ocean has, from about -5 to 5.
    """
    if n == 0:
        return []
    return [float(x) for x in np.round(sample_window("elnino", rng, n)) - 24.0]


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_small_random_lists_match_brute_force(seed: int) -> None:
    rng = data_rng(f"days-until-higher-p01:small:{seed}")
    values = _random_readings(rng, int(rng.integers(0, 14)), low=-4, high=6)
    _verify(values)


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_medium_random_lists_with_ties_match_brute_force(seed: int) -> None:
    rng = data_rng(f"days-until-higher-p01:medium:{seed}")
    values = _random_readings(rng, int(rng.integers(50, 260)), low=0, high=8)
    _verify(values)


def test_empty_and_single_reading() -> None:
    _verify([])
    _verify([4.0])


def test_strictly_decreasing_readings_have_no_higher_reading_after_them() -> None:
    values = [9.0, 7.0, 4.0, 1.0]
    got, problem = _call(values)
    if problem is None and got != [0, 0, 0, 0]:
        problem = (
            "a falling run has no strictly larger reading after any of its entries: every entry "
            "is 0; "
            "a leftover value or -1 is wrong"
        )
    if problem is not None:
        _fail(problem)


def test_equal_runs_are_not_higher_than_each_other() -> None:
    values = [2.0, 2.0, 2.0, 3.0]
    got, problem = _call(values)
    if problem is None and got != [3, 2, 1, 0]:
        problem = (
            "a later reading equal to this one ended its wait: only a strictly larger reading "
            "ends the "
            "wait, so use > and not >="
        )
    if problem is not None:
        _fail(problem)


def test_one_higher_reading_resolves_a_whole_falling_run() -> None:
    values = [5.0, 4.0, 3.0, 2.0, 1.0, 6.0]
    got, problem = _call(values)
    if problem is None and got != [5, 4, 3, 2, 1, 0]:
        problem = (
            "readings still waiting when a higher reading arrived were left unresolved: pop every "
            "waiting "
            "reading that is lower, not just one (a while loop, not an if)"
        )
    if problem is not None:
        _fail(problem)


def test_negative_readings_compare_by_value_not_magnitude() -> None:
    values = [-1.0, -2.0, 0.0]
    got, problem = _call(values)
    if problem is None and got != [2, 1, 0]:
        problem = (
            "negative readings were compared by magnitude or sign: compare their signed values"
        )
    if problem is not None:
        _fail(problem)


def test_input_list_is_not_mutated() -> None:
    values = [5.0, 3.0, 8.0, 8.0, 2.0, 9.0]
    before = list(values)
    _, problem = _call(values)
    if problem is not None:
        _fail(problem)
    if values != before:
        _fail(
            "the input list was changed: do not append a sentinel or pop from values; keep the "
            "waiting "
            "indices in a separate stack"
        )


def test_entries_are_python_ints_for_fractional_readings() -> None:
    got, problem = _call([0.5, 0.25, 0.75, -3.5])
    if problem is None and not all(type(entry) is int for entry in got):
        problem = "each entry must be a plain Python int, not a float or a NumPy scalar"
    if problem is None and got != [2, 1, 0, 0]:
        problem = "the fractional readings are compared wrongly: compare the values as they are"
    if problem is not None:
        _fail(problem)


_CHILD = f"""
from submission import days_until_higher
n = {LARGE_N}
values = [float(n - i) for i in range(n)] + [float(n + 1)]
try:
    res = days_until_higher(values)
except Exception as exc:
    print("RAISED", type(exc).__name__)
else:
    print("RESULT", len(res), res[0], res[-1], sum(res), all(type(x) is int for x in res))
"""


def test_large_list_of_200000_readings_finishes_within_the_time_limit() -> None:
    # Analytic answer: a falling run of n readings, then one reading above all of them. Each entry i
    # waits n - i steps for the last reading, and the last entry has no answer.
    n = LARGE_N
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
            f"too slow on {n} readings: the function did not finish in {TIME_LIMIT_S:g} s. "
            f"Checking every "
            "later position for each reading is the usual cause; one forward pass with a stack is "
            "O(n)"
        )
    lines = proc.stdout.strip().splitlines()
    line = lines[-1] if lines else ""
    if line.startswith("RAISED"):
        pytest.fail(f"your function raised on the large input: {line[len('RAISED ') :]}")
    fields = line.split()
    if not fields or fields[0] != "RESULT":
        pytest.fail("large run returned no result: check runtime errors and return type")
    length, first, last, total, all_int = fields[1:]
    if int(length) != n + 1:
        _fail("the result must have one entry per reading: the same length as the input list")
    if all_int != "True":
        _fail("each entry must be a plain Python int, not a float or a NumPy scalar")
    if int(first) != n or int(last) != 0:
        _fail(
            "an entry on the large input is wrong: each reading waits for the one above all others"
        )
    if int(total) != n * (n + 1) // 2:
        _fail(
            "the entries on the large input do not add up to the expected total: check the "
            "distances"
        )
