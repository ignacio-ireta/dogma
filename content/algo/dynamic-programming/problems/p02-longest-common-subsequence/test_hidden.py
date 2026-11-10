"""Hidden tests: a subset brute force for short strings, an independent suffix-based table for
longer ones, a witness check, and the edge families named in problem.md.

Every failure names the kind of mistake and carries no hidden input or answer.
"""

from __future__ import annotations

import random
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import combinations
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import longest_common_subsequence

LONG_LENGTH = 1200
# The reference answers the 1,200-character case in well under a second; the budget leaves room.
LONG_BUDGET_S = 10.0


class _TooSlow(BaseException):
    """Raised inside the learner's code when the time box expires."""


@contextmanager
def _time_box(seconds: float) -> Iterator[None]:
    """Interrupt running code after ``seconds``. Only the main thread can use signals."""
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


def _answer(a: Any, b: Any, limit: float = 5.0) -> Any:
    try:
        with _time_box(limit):
            return longest_common_subsequence(a, b)
    except _TooSlow:
        pass  # reported below, outside this except block, so no chained exception is shown
    except RecursionError:
        pytest.fail(
            "RecursionError on a long input: a recursion that goes one level per character is too "
            "deep for Python's default limit; fill the table in loops"
        )
    except Exception as exc:
        pytest.fail(
            f"longest_common_subsequence raised {type(exc).__name__} on a valid input: {exc}"
        )
    pytest.fail(
        "a call did not finish in time: a search over every subset or every pair of "
        "prefixes without a table is the usual cause; one table of prefix lengths is quadratic"
    )


def _is_subsequence(part: str, whole: str) -> bool:
    remaining = iter(whole)
    return all(character in remaining for character in part)


def _brute_length(a: str, b: str) -> int:
    """Independent oracle: the longest subset of positions of a that spells a subsequence of b."""
    best = 0
    for size in range(len(a), 0, -1):
        for positions in combinations(range(len(a)), size):
            candidate = "".join(a[p] for p in positions)
            if _is_subsequence(candidate, b):
                return size
    return best


def _suffix_length(a: str, b: str) -> int:
    """Independent oracle: a suffix-based table, filled from the end, two rows at a time."""
    n, m = len(a), len(b)
    below = [0] * (m + 1)  # row for a[i+1:], initially the empty suffix of a
    for i in range(n - 1, -1, -1):
        row = [0] * (m + 1)
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                row[j] = below[j + 1] + 1
            else:
                row[j] = max(below[j], row[j + 1])
        below = row
    return below[0]


MISTAKE = (
    "the length differs from the reference: check the base row and column, the match transition "
    "and the max over the two neighbours"
)


def _check(a: str, b: str, expected: int) -> None:
    result = _answer(a, b)
    if type(result) is not tuple or len(result) != 2:
        pytest.fail("return a (length, witness) tuple of two values")
    length, witness = result
    if type(length) is not int or type(witness) is not str:
        pytest.fail("return (int, str): the length must be an int and the witness a str")
    if length != expected:
        pytest.fail(MISTAKE)
    if len(witness) != length:
        pytest.fail("the witness must have exactly the returned length: check the walk back")
    if not _is_subsequence(witness, a) or not _is_subsequence(witness, b):
        pytest.fail(
            "the witness is not a subsequence of both strings: keep matched characters in order, "
            "and take them only in a walk back over matching positions"
        )


@settings(max_examples=120, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_property_length_and_witness_on_random_short_strings(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic strings, not security.
    alphabet = rng.choice(["ab", "abc", "acgt"])
    a = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 13)))
    b = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 13)))
    _check(a, b, _suffix_length(a, b))


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_tiny_strings_match_the_subset_brute_force(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic strings, not security.
    a = "".join(rng.choice("xyz") for _ in range(rng.randrange(0, 8)))
    b = "".join(rng.choice("xyz") for _ in range(rng.randrange(0, 8)))
    _check(a, b, _brute_length(a, b))


def test_empty_strings_give_zero_and_the_empty_witness() -> None:
    for a, b in (("", ""), ("", "xyz"), ("abc", "")):
        if _answer(a, b) != (0, ""):
            pytest.fail('an empty string has no common subsequence: return (0, "")')


def test_identical_strings_return_the_whole_string() -> None:
    if _answer("kitten", "kitten") != (6, "kitten"):
        pytest.fail("identical strings share themselves: the length is the whole length")


def test_repeated_characters_count_each_match_once() -> None:
    if _answer("aaaa", "aa") != (2, "aa"):
        pytest.fail("with aaaa and aa only two characters can be matched, so the length is 2")


def test_order_matters_not_just_the_letter_counts() -> None:
    _check("ab", "ba", 1)


def test_substring_trap_is_not_a_contiguous_block() -> None:
    length, _ = _answer("abcde", "aXcXe")
    if length != 3:
        pytest.fail("the answer is a subsequence, not a contiguous block: the length must be 3")


def test_long_strings_finish_within_the_time_limit() -> None:
    rng = random.Random(4242)  # noqa: S311 -- seeded synthetic strings, not security.
    a = "".join(rng.choice("acgt") for _ in range(LONG_LENGTH))
    b = "".join(rng.choice("acgt") for _ in range(LONG_LENGTH))
    result = _answer(a, b, LONG_BUDGET_S)
    if type(result) is not tuple or result[0] != _suffix_length(a, b):
        pytest.fail(MISTAKE)
    if not _is_subsequence(result[1], a) or not _is_subsequence(result[1], b):
        pytest.fail("the witness is not a subsequence of both long strings")


def test_return_type_is_an_int_and_a_str() -> None:
    length, witness = _answer("abcd", "bd")
    if type(length) is not int or type(witness) is not str:
        pytest.fail("return (int, str): a Python int length and a Python str witness")


@pytest.mark.parametrize(
    ("a", "b"),
    [(b"abc", "abc"), ("abc", None), (["a"], "a"), (3, "abc")],
    ids=["bytes-first", "none-second", "list-first", "int-first"],
)
def test_non_str_input_raises_type_error(a: Any, b: Any) -> None:
    try:
        longest_common_subsequence(a, b)
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"a non-str input must raise TypeError, instead it raised {type(exc).__name__}")
    pytest.fail("a non-str input was accepted: check that both arguments are Python str")


def test_characters_are_compared_by_equality_not_by_case() -> None:
    # "A" and "a" are different characters, so only "b" is shared: the length is 1.
    if _answer("Ab", "ab") != (1, "b"):
        pytest.fail(
            "characters are compared by equality: letters that differ only in case "
            "are different characters"
        )


def test_second_argument_must_be_a_str_too() -> None:
    # A check on the first argument alone lets a list or bytes second argument through,
    # because len() and indexing both work on them.
    for b in (["a", "b"], b"ab"):
        try:
            longest_common_subsequence("ab", b)  # type: ignore[arg-type]
        except TypeError:
            continue
        pytest.fail(
            "a list or bytes second argument must raise TypeError: check both arguments are str"
        )
