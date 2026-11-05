"""Hidden tests: a stable-sort oracle, laziness through read counters, labels, errors and speed.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

from __future__ import annotations

import itertools
import signal
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import merge_sorted_streams

READ_CAP = 20_000  # total reads across the counted streams before a learner is stopped
LARGE_STREAMS = 5_000
LARGE_LENGTH = 20
LARGE_TIME_LIMIT_S = 6.0
ORDER_MISTAKE = (
    "the merged order differs from the stable reference: output by key, and among equal keys "
    "put the lower stream index first, keeping each stream's own order"
)


class _TooGreedy(BaseException):
    """Raised inside a counted stream when far more is read than any lazy merge needs."""


class _TooSlow(BaseException):
    """Raised inside the learner's call when it runs past its time box."""


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
    raise AssertionError(why)


class _Meter:
    """One read counter shared by every stream in a test."""

    def __init__(self) -> None:
        self.total = 0


class _CountedStream:
    """An iterable that counts its own reads and stops the test if the total is absurd."""

    def __init__(self, items: Iterable[Any], meter: _Meter) -> None:
        self._iterator = iter(items)
        self._meter = meter
        self.reads = 0

    def __iter__(self) -> _CountedStream:
        return self

    def __next__(self) -> Any:
        self.reads += 1
        self._meter.total += 1
        if self._meter.total > READ_CAP:
            raise _TooGreedy
        return next(self._iterator)


def _collect(merge: Any, limit: int | None = None) -> list[tuple[Any, Any]]:
    """Take up to ``limit`` pairs (all if None). A crash is a contract mistake, named here."""
    out: list[tuple[Any, Any]] = []
    try:
        iterator = iter(merge)
        while limit is None or len(out) < limit:
            out.append(next(iterator))
    except StopIteration:
        pass
    except _TooGreedy:
        _fail(
            "the merge read far more than the caller asked for: read a stream only when its "
            "head has been output"
        )
    except _TooSlow:
        _fail("too slow on a large merge: keep a heap of the stream heads, not a scan of them")
    except Exception as exc:
        _fail(
            f"your merge raised {type(exc).__name__}: check empty streams, the heap entries and "
            "the refill step"
        )
    return out


def _stable_reference(streams: list[list[tuple[Any, Any]]]) -> list[tuple[Any, Any]]:
    """Independent answer: concatenate in stream order, then a stable sort by key."""
    return sorted(itertools.chain.from_iterable(streams), key=lambda pair: pair[0])


def _random_streams(rng: np.random.Generator) -> list[list[tuple[Any, Any]]]:
    streams = []
    for index in range(int(rng.integers(0, 7))):
        keys = sorted(int(v) for v in rng.integers(0, 6, size=int(rng.integers(0, 11))))
        streams.append([(key, (index, position)) for position, key in enumerate(keys)])
    return streams


@pytest.mark.parametrize("seed", range(60), ids=[f"random-{s}" for s in range(60)])
def test_matches_a_stable_sort_of_the_concatenated_streams(seed: int) -> None:
    streams = _random_streams(np.random.default_rng(seed))
    want = _stable_reference(streams)  # computed before the learner runs
    got = _collect(merge_sorted_streams(streams))
    assert got == want, ORDER_MISTAKE


def test_equal_keys_come_out_in_stream_order_even_when_labels_sort_the_other_way() -> None:
    streams = [[(5, "z")], [(5, "a")], [(5, "m")]]
    got = _collect(merge_sorted_streams(streams))
    assert [label for _, label in got] == ["z", "a", "m"], (
        "equal keys must follow stream index, without ordering the payload labels"
    )


def test_labels_are_never_compared() -> None:
    streams = [[(1, {"id": "x"})], [(1, {"id": "y"})]]
    got = _collect(merge_sorted_streams(streams))
    assert [key for key, _ in got] == [1, 1], (
        "labels were compared: dictionaries cannot be ordered, so compare only the key and the "
        "stream index"
    )


def test_nothing_is_read_when_the_merge_is_created() -> None:
    meter = _Meter()
    streams = [_CountedStream([(1, "a"), (3, "b")], meter), _CountedStream([(2, "c")], meter)]
    merge = merge_sorted_streams(streams)
    assert meter.total == 0, (
        "calling merge_sorted_streams must read nothing: build a generator, and read on next()"
    )
    assert [label for _, label in _collect(merge)] == ["a", "c", "b"], (
        "equal keys must preserve stream-index order and each stream's own order"
    )


def test_infinite_streams_merge_lazily_and_read_only_what_is_needed() -> None:
    meter = _Meter()
    evens = _CountedStream(((k, (0, k)) for k in itertools.count(0, 2)), meter)
    odds = _CountedStream(((k, (1, k)) for k in itertools.count(1, 2)), meter)
    finite = _CountedStream([(5, (2, 0)), (6, (2, 1))], meter)
    streams = [evens, odds, finite]
    got = _collect(merge_sorted_streams(streams), limit=12)
    assert [key for key, _ in got] == [0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9], (
        "the first twelve pairs of the merge are wrong: check the key order and the tie rule"
    )
    taken = [0, 0, 0]
    for _, (stream, _) in got:
        taken[stream] += 1
    for stream, counted in zip(streams, taken, strict=True):
        assert counted >= 0 and stream.reads <= counted + 1, (
            "a stream was read more than one pair beyond what was taken: read a stream only when "
            "its head has been output"
        )


@pytest.mark.parametrize("seed", range(12), ids=[f"budget-{s}" for s in range(12)])
def test_each_stream_is_read_at_most_one_pair_beyond_what_was_taken(seed: int) -> None:
    rng = np.random.default_rng(seed)
    raw = _random_streams(rng)
    total = sum(len(s) for s in raw)
    for count in range(total + 1):
        meter = _Meter()
        counted = [_CountedStream(s, meter) for s in raw]
        got = _collect(merge_sorted_streams(counted), limit=count)
        taken = [0] * len(raw)
        for _, (stream, _) in got:
            taken[stream] += 1
        for stream, handle in enumerate(counted):
            if handle.reads > taken[stream] + 1:
                _fail(
                    "a stream was read further ahead than needed: a lazy merge holds at most one "
                    "pair per stream, so read the next pair only after the previous one is output"
                )


def test_empty_and_single_streams() -> None:
    assert _collect(merge_sorted_streams([])) == [], "no streams gives an empty merge"
    assert _collect(merge_sorted_streams([[], []])) == [], "only empty streams gives an empty merge"
    assert _collect(merge_sorted_streams([[(1, "a")]])) == [(1, "a")], (
        "one stream comes out unchanged"
    )


def test_generator_streams_are_accepted() -> None:
    streams = [((k, "g0") for k in (1, 3, 5)), ((k, "g1") for k in (2, 3, 6))]
    got = _collect(merge_sorted_streams(streams))
    assert [key for key, _ in got] == [1, 2, 3, 3, 5, 6], (
        "generator streams must work as well as lists: merge them as they come"
    )
    assert [label for _, label in got][2:4] == ["g0", "g1"], (
        "equal keys must come out in stream-index order"
    )


def test_the_result_is_an_iterator_not_a_list() -> None:
    merge = merge_sorted_streams([[(1, "a")], [(2, "b")]])
    assert iter(merge) is merge, "return an iterator (a generator), so the merge is lazy"


def test_keys_may_be_negative_floats() -> None:
    streams = [[(-2.5, "a"), (0.0, "b")], [(-2.5, "c"), (1.25, "d")]]
    got = _collect(merge_sorted_streams(streams))
    assert got == [(-2.5, "a"), (-2.5, "c"), (0.0, "b"), (1.25, "d")], (
        "keys compare as numbers, including negative floats, with ties in stream order"
    )


def test_inputs_are_not_mutated() -> None:
    streams = [[(1, "a"), (4, "b")], [(2, "c"), (4, "d")]]
    before = [list(s) for s in streams]
    _collect(merge_sorted_streams(streams))
    assert streams == before, "a stream was changed: read it, never edit it"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(
        st.lists(st.integers(-5, 5), min_size=0, max_size=8).map(sorted),
        min_size=0,
        max_size=5,
    )
)
def test_output_is_sorted_and_keeps_every_pair(keys_per_stream: list[list[int]]) -> None:
    streams = [[(k, (i, p)) for p, k in enumerate(keys)] for i, keys in enumerate(keys_per_stream)]
    got = _collect(merge_sorted_streams(streams))
    got_keys = [key for key, _ in got]
    assert got_keys == sorted(got_keys), "the output keys must be non-decreasing"
    assert sorted(got, key=repr) == sorted(itertools.chain.from_iterable(streams), key=repr), (
        "every input pair must come out exactly once"
    )


def test_large_merge_finishes_within_the_time_limit_and_matches_the_stable_sort() -> None:
    rng = np.random.default_rng(21)
    streams = []
    for index in range(LARGE_STREAMS):
        keys = sorted(int(v) for v in rng.integers(0, 10_000, size=LARGE_LENGTH))
        streams.append([(key, (index, position)) for position, key in enumerate(keys)])
    want = _stable_reference(streams)
    merge = merge_sorted_streams(streams)
    try:
        with _time_box(LARGE_TIME_LIMIT_S):
            got = list(merge)
    except _TooSlow:
        _fail(
            "too slow on a large merge: keep a heap of the stream heads, so each output costs "
            "O(log k), not a scan of all k heads"
        )
    except Exception as exc:
        _fail(f"your merge raised {type(exc).__name__} on a large input: check the refill step")
    assert got == want, ORDER_MISTAKE
