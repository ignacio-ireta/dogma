"""Hidden tests: a two-pass oracle of the protocol, uniformity, one-pass reading, and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or an expected
answer.
"""

from __future__ import annotations

import functools
import itertools
import re
import tracemalloc
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import reservoir_sample

from dogma_data import load_text


class _RecordingRng:
    """Wraps a real Generator and records each ``integers`` call as a (low, high) pair."""

    def __init__(self, seed):
        self._generator = np.random.default_rng(seed)
        self.calls = []

    def integers(self, low, high=None):
        if high is None:
            low, high = 0, low
        self.calls.append((low, high))
        return self._generator.integers(low, high)


def _run(call, mistake):
    """Call the learner on valid inputs. An exception becomes a failure that names the mistake."""
    try:
        return call()
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _reference(items, k, seed):
    """Independent two-pass version of the protocol: draw all slots first, then apply them."""
    items = list(items)
    reservoir = items[:k]
    rng = np.random.default_rng(seed)
    slots = [(i, int(rng.integers(0, i + 1))) for i in range(k, len(items))]
    for i, j in slots:
        if j < k:
            reservoir[j] = items[i]
    return reservoir


@functools.cache
def _words():
    """A real stream: the words of Frankenstein in reading order, repeats included."""
    return re.findall(r"[A-Za-z']+", load_text("gutenberg_frankenstein"))


@functools.cache
def _distinct_words():
    """The same stream with each word kept at its first appearance only."""
    return list(dict.fromkeys(_words()))


def _sample(n, k, seed):
    """Sample k of the first n distinct real words; return their positions in the stream."""
    stream = _distinct_words()[:n]
    kept = _run(
        lambda: reservoir_sample(stream, k, np.random.default_rng(seed)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    position = {word: index for index, word in enumerate(stream)}
    if any(word not in position for word in kept):
        pytest.fail("the sample holds an item that is not in the stream")
    return [position[word] for word in kept]


@settings(max_examples=150, deadline=None, report_multiple_bugs=False)
@given(
    n=st.integers(min_value=0, max_value=60),
    k=st.integers(min_value=0, max_value=12),
    seed=st.integers(min_value=0, max_value=10**6),
)
def test_matches_a_two_pass_reference_of_the_same_protocol(n, k, seed):
    # A real stream: the first n words of Frankenstein, repeats included.
    items = _words()[:n]
    got = _run(
        lambda: reservoir_sample(items, k, np.random.default_rng(seed)),
        "valid inputs must not raise: a slot index must stay below k; reservoir_sample raised",
    )
    assert got == _reference(items, k, seed), (
        "the sample differs from the protocol: fill the first k slots, then for each later item "
        "draw j = rng.integers(0, i + 1) and replace slot j only when j < k"
    )


@pytest.mark.parametrize(
    ("n", "k"),
    [(10, 3), (7, 1), (5, 5), (4, 6), (30, 4)],
    ids=["case-a", "case-b", "case-c", "case-d", "case-e"],
)
def test_draws_are_one_integers_call_per_item_after_the_first_k(n, k):
    rng = _RecordingRng(0)
    _run(
        lambda: reservoir_sample(_words()[:n], k, rng),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert rng.calls == [(0, i + 1) for i in range(k, n)], (
        "the draws must be one rng.integers(0, i + 1) per item after the first k, and no others"
    )


def test_each_item_is_kept_with_probability_k_over_n():
    n, k, runs = 10, 3, 3000
    counts = np.zeros(n)
    for seed in range(runs):
        counts[_sample(n, k, seed)] += 1
    p_value = stats.chisquare(counts, np.full(n, runs * k / n)).pvalue
    assert p_value > 0.001, (
        "items are not kept with equal probability k / n: check the replacement rule and the "
        "range of the draw"
    )


def test_every_subset_of_size_k_is_equally_likely():
    n, k, runs = 5, 2, 5000
    counts = Counter(tuple(sorted(_sample(n, k, seed))) for seed in range(runs))
    subsets = list(itertools.combinations(range(n), k))
    observed = np.array([counts[subset] for subset in subsets])
    p_value = stats.chisquare(observed, np.full(len(subsets), runs / len(subsets))).pvalue
    assert p_value > 0.001, (
        "some size-k subsets are more likely than others: the sample is not uniform"
    )


def test_generator_input_is_read_to_its_end_and_matches_the_list_result():
    def stream():
        yield from _words()[:25]

    source = stream()
    from_generator = _run(
        lambda: reservoir_sample(source, 4, np.random.default_rng(9)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    from_list = _run(
        lambda: reservoir_sample(_words()[:25], 4, np.random.default_rng(9)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert next(source, "end") == "end", (
        "the stream must be read to its end: a one-pass sample needs every item"
    )
    assert from_generator == from_list, (
        "a generator must give the same sample as the list with the same seed"
    )


def test_stream_shorter_than_k_and_empty_stream_return_what_there_is():
    short = _run(
        lambda: reservoir_sample(["p", "q"], 5, np.random.default_rng(0)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert short == ["p", "q"], "a stream shorter than k is returned whole, in arrival order"
    empty = _run(
        lambda: reservoir_sample([], 3, np.random.default_rng(0)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert empty == [], "an empty stream gives an empty sample"


def test_memory_stays_bounded_by_k_on_a_long_stream():
    words = _words()

    def long_stream():
        # Three hundred thousand real words: the book read again and again, one word at a
        # time. A list of that many references alone is over two megabytes.
        return (words[index % len(words)] for index in range(300_000))

    tracemalloc.start()
    try:
        sample = _run(
            lambda: reservoir_sample(long_stream(), 10, np.random.default_rng(1)),
            "valid inputs must not raise: reservoir_sample raised",
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(sample) == 10, "the sample must hold exactly k items for a long stream"
    assert peak < 1024 * 1024, (
        "the stream is being stored: keep only the k reservoir items while streaming"
    )


def test_k_zero_returns_an_empty_list():
    assert (
        _run(
            lambda: reservoir_sample(range(6), 0, np.random.default_rng(0)),
            "valid inputs must not raise: reservoir_sample raised",
        )
        == []
    ), "k = 0 should return an empty list"


def test_negative_k_raises_value_error():
    _raises_value_error(
        lambda: reservoir_sample(range(5), -1, np.random.default_rng(0)),
        "k must be non-negative: raise ValueError when k is negative",
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    items=st.lists(st.integers(min_value=-5, max_value=5), max_size=40),
    k=st.integers(min_value=0, max_value=12),
    seed=st.integers(min_value=0, max_value=10**6),
)
def test_sample_has_min_k_n_items_drawn_from_the_stream(items, k, seed):
    got = _run(
        lambda: reservoir_sample(items, k, np.random.default_rng(seed)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert isinstance(got, list), "return a list of the sampled items"
    assert len(got) == min(k, len(items)), "the sample must hold min(k, n) items"
    assert not (Counter(got) - Counter(items)), "every sampled item must come from the stream"


def test_input_list_is_not_mutated():
    items = list(range(20))
    before = list(items)
    _run(
        lambda: reservoir_sample(items, 4, np.random.default_rng(3)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert items == before, (
        "the input list was modified: keep the reservoir separate from the stream"
    )


def test_list_items_come_back_unchanged():
    items = [[i, i + 1] for i in range(12)]
    before = [list(item) for item in items]
    _run(
        lambda: reservoir_sample(items, 3, np.random.default_rng(5)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert items == before, (
        "the stream's items were modified: the sample must hold the items unchanged"
    )
