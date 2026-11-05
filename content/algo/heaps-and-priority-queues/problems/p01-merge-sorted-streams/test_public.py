"""Public tests: the examples from problem.md, empty input, and the stable tie order.

The learner's code is imported from submission.py. Tests never import solution.py.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from submission import merge_sorted_streams

READ_CAP = 5_000  # reads across the infinite streams before a learner is stopped


class _TooGreedy(BaseException):
    """Raised when the merge reads far more than the caller asked for."""


class _Capped:
    """An infinite or long stream that stops the test once the shared read cap is exceeded."""

    def __init__(self, items: Iterable[Any], counter: list[int]) -> None:
        self._iterator: Iterator[Any] = iter(items)
        self._counter = counter

    def __iter__(self) -> _Capped:
        return self

    def __next__(self) -> Any:
        self._counter[0] += 1
        if self._counter[0] > READ_CAP:
            raise _TooGreedy
        return next(self._iterator)


def _collect(merge: Any) -> list[tuple[Any, Any]]:
    try:
        return list(merge)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the heap entries and the refill step"
        )


def test_two_streams_merge_in_key_order_with_ties_by_stream() -> None:
    got = _collect(merge_sorted_streams([[(1, "a"), (4, "b")], [(2, "c"), (4, "d")]]))
    assert len(got) == 4, "every input pair must come out exactly once: do not drop pairs"
    assert got == [(1, "a"), (2, "c"), (4, "b"), (4, "d")], (
        "output must be sorted by key; the tied keys 4 come out in stream order: 'b' is in "
        "stream 0 and 'd' is in stream 1"
    )


def test_infinite_streams_are_consumed_only_as_far_as_the_caller_asks() -> None:
    counter = [0]
    evens = _Capped(((k, "even") for k in itertools.count(0, 2)), counter)
    odds = _Capped(((k, "odd") for k in itertools.count(1, 2)), counter)
    try:
        got = list(itertools.islice(merge_sorted_streams([evens, odds]), 6))
    except _TooGreedy:
        pytest.fail(
            "the merge read far more than the caller asked: read a stream only when its head "
            "has been output, and never build a whole stream in memory"
        )
    except Exception as exc:
        pytest.fail(f"valid input raised {type(exc).__name__}: check the refill step")
    assert [key for key, _ in got] == [0, 1, 2, 3, 4, 5], (
        "with infinite inputs the merge must yield the smallest pairs first"
    )


def test_empty_streams_contribute_nothing() -> None:
    assert _collect(merge_sorted_streams([[], iter([(3, "x")]), ()])) == [(3, "x")], (
        "an empty stream must be skipped, not treated as an error or as the end of the merge"
    )
    assert _collect(merge_sorted_streams([])) == [], "no streams at all gives an empty result"
