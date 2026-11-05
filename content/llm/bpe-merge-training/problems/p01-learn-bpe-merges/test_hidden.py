"""Hidden tests: a naive trainer as the oracle, the tie-break, weights, marker and type rules.

Failure messages name the kind of mistake and carry no hidden word table or merge list.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, NoReturn

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import train_bpe_merges

END = "</w>"


def _fail(why: str) -> NoReturn:
    """The one place these tests raise, so every mistake is reported with a single named line."""
    raise AssertionError(why)


def _expect_error(exc_type: type[BaseException], make: Callable[[], Any], message: str) -> None:
    """Pass when make() raises exc_type. Otherwise fail with message, naming what happened."""
    try:
        make()
    except exc_type:
        return
    except Exception as exc:
        wrong = type(exc).__name__
    else:
        _fail(f"{message}: nothing was raised")
    _fail(f"{message}: got {wrong} instead")


def _naive_train(word_counts: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Independent oracle. It rebuilds every count from the raw symbol lists at each step.

    The word table is rewritten as a list of (count, symbols) rows. Merging is a while loop that
    looks for the pair at the current position, which is a different code path from the solution's
    list comprehension.
    """
    rows = [(count, [*word, END]) for word, count in word_counts.items()]
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        tally: dict[tuple[str, str], int] = {}
        for count, symbols in rows:
            pos = 0
            while pos + 1 < len(symbols):
                key = (symbols[pos], symbols[pos + 1])
                tally[key] = tally.get(key, 0) + count
                pos += 1
        if not tally:
            break
        ordered = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
        pick = ordered[0][0]
        merges.append(pick)
        new_rows = []
        for count, symbols in rows:
            merged: list[str] = []
            pos = 0
            while pos < len(symbols):
                if pos + 1 < len(symbols) and (symbols[pos], symbols[pos + 1]) == pick:
                    merged.append(symbols[pos] + symbols[pos + 1])
                    pos += 2
                else:
                    merged.append(symbols[pos])
                    pos += 1
            new_rows.append((count, merged))
        rows = new_rows
    return merges


def _random_table(rng: np.random.Generator) -> dict[str, int]:
    table: dict[str, int] = {}
    for _ in range(int(rng.integers(0, 9))):
        length = int(rng.integers(1, 6))
        word = "".join("abc"[int(rng.integers(3))] for _ in range(length))
        table[word] = int(rng.integers(1, 7))
    return table


def _check_against_oracle(table: dict[str, int], num_merges: int) -> None:
    expected = _naive_train(table, num_merges)
    try:
        got = train_bpe_merges(table, num_merges)
    except Exception as exc:
        error: str | None = type(exc).__name__
    else:
        error = None
    if error is not None:
        _fail(f"your trainer raised {error} on a valid table: check the count loop")
    if type(got) is not list or any(type(m) is not tuple for m in got):
        _fail("return a list of (left, right) tuples, one per merge, in the order learned")
    if got != expected:
        _fail(
            "merge list differs from the naive trainer: check the tie-break (smallest pair), the "
            "weights (each count multiplied by the word count), the end-of-word marker, and the "
            "left-to-right merge without overlaps"
        )


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_tables_match_the_naive_trainer(seed: int) -> None:
    rng = np.random.default_rng(seed)
    _check_against_oracle(_random_table(rng), int(rng.integers(0, 15)))


@settings(max_examples=40, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_order_of_the_table_does_not_change_the_merges(seed: int) -> None:
    rng = np.random.default_rng(seed)
    table = _random_table(rng)
    reversed_table = dict(reversed(list(table.items())))
    try:
        forward = train_bpe_merges(table, 8)
        backward = train_bpe_merges(reversed_table, 8)
    except Exception as exc:
        error: str | None = type(exc).__name__
    else:
        error = None
    if error is not None:
        _fail(f"your trainer raised {error} on a valid table")
    if forward != backward:
        _fail(
            "the merges depend on the order of the dict: the tie-break must not use insertion order"
        )


def test_tie_between_two_pairs_goes_to_the_smaller_pair_in_both_insertion_orders() -> None:
    # (a, b) and (b, a) each count 1 here, so the smaller pair (a, b) must be learned first.
    table_a = {"ab": 1, "ba": 1}
    table_b = {"ba": 1, "ab": 1}
    for table in (table_a, table_b):
        got = train_bpe_merges(table, 1)
        if got[0] != ("a", END):
            _fail(
                "a tie must go to the smallest pair: ('a', '</w>') sorts before ('a', 'b') "
                "because '<' comes before 'b'"
            )


def test_end_of_word_marker_takes_part_in_merges() -> None:
    got = train_bpe_merges({"ab": 1, "ba": 1}, 2)
    if got != [("a", END), ("a", "b")]:
        _fail("the end-of-word marker must be a symbol that merges with the last character")


def test_overlapping_occurrences_each_count_toward_the_pair() -> None:
    if train_bpe_merges({"aaa": 1, "bb": 1}, 1) != [("a", "a")]:
        _fail("overlapping occurrences must each count: 'aaa' holds two (a, a) pairs")


def test_overlapping_merge_is_applied_left_to_right_once() -> None:
    # 'aaa' merges its left two a's first. Then the leftover a pairs with the marker, so the
    # second merge is (a, </w>). A right-to-left scan would merge the right two a's instead.
    got = train_bpe_merges({"aaa": 1}, 2)
    if got != [("a", "a"), ("a", END)]:
        _fail(
            "overlapping merges must be applied left to right: 'aaa' becomes 'aa' and 'a', and "
            "the leftover 'a' then pairs with the end-of-word marker"
        )


def test_word_counts_weight_the_pairs() -> None:
    if train_bpe_merges({"ab": 1, "cd": 2}, 1) != [("c", "d")]:
        _fail(
            "each pair must be weighted by its word count: 'cd' appears twice, so (c, d) must win "
            "over the unweighted smallest pair"
        )


def test_learning_stops_when_every_word_is_a_single_symbol() -> None:
    got = train_bpe_merges({"a": 3, "b": 1}, 10)
    if got != [("a", END), ("b", END)]:
        _fail(
            "when no adjacent pair is left, the trainer must stop and return the merges so far: "
            "after two merges each word is one symbol"
        )


def test_single_character_words_with_one_merge_available() -> None:
    if train_bpe_merges({"a": 3}, 2) != [("a", END)]:
        _fail("a table of one one-character word has one merge at most, then learning must stop")


def test_requested_merges_longer_than_the_table_return_fewer_merges() -> None:
    got = train_bpe_merges({"ab": 2}, 50)
    if len(got) > 2:
        _fail("the number of merges is bounded by the table: stop when no pair is left")


def test_large_counts_keep_exact_pair_totals() -> None:
    # 10**18 and 10**18 + 1 round to the same float, so a float count would tie (a, b) with (a, c).
    got = train_bpe_merges({"ab": 10**18, "ac": 10**18 + 1}, 1)
    if got != [("a", "c")]:
        _fail(
            "pair counts must stay exact integers above 2**53: a float count loses the difference"
        )


def test_merges_are_returned_as_tuples_of_two_strings() -> None:
    got = train_bpe_merges({"abab": 2, "ab": 1}, 3)
    for item in got:
        if type(item) is not tuple or len(item) != 2 or any(type(s) is not str for s in item):
            _fail("each merge must be a tuple of two str: (left, right), not the merged string")
    if got and got[0] != ("a", "b"):
        _fail("the first merge of 'abab' and 'ab' is (a, b), which counts 4 for this table")


def test_input_table_is_not_mutated() -> None:
    table = {"low": 5, "lower": 2}
    before = copy.deepcopy(table)
    train_bpe_merges(table, 3)
    if table != before:
        _fail("word_counts was changed: do not pop, sort or rewrite the input dict")


def test_non_dict_table_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: train_bpe_merges([("low", 1)], 1),  # type: ignore[arg-type]  # invalid table
        "a list of pairs is not a word-count dict and must raise TypeError",
    )


def test_bool_count_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: train_bpe_merges({"low": True}, 1),  # invalid count
        "a bool count must raise TypeError: check type(count) is int",
    )


def test_zero_count_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: train_bpe_merges({"low": 0}, 1),
        "a count below 1 must raise ValueError",
    )


def test_empty_word_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: train_bpe_merges({"": 1}, 1),
        "an empty word must raise ValueError",
    )


def test_word_with_whitespace_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: train_bpe_merges({"two words": 1}, 1),
        "a word with whitespace must raise ValueError: split the text into words first",
    )


def test_word_containing_the_marker_text_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: train_bpe_merges({"a</w>": 1}, 1),
        "a word containing the end-of-word marker text must raise ValueError",
    )


def test_bool_num_merges_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: train_bpe_merges({"low": 1}, True),  # invalid num_merges
        "a bool num_merges must raise TypeError",
    )


def test_negative_num_merges_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: train_bpe_merges({"low": 1}, -1),
        "a negative num_merges must raise ValueError",
    )


def test_tuple_key_in_word_counts_raises_type_error() -> None:
    # Every key must be a str word. A tuple of symbols is hashable, so only a type check rejects it.
    _expect_error(
        TypeError,
        lambda: train_bpe_merges({("l", "o"): 1}, 1),  # type: ignore[dict-item]  # invalid key
        "a word-count key that is not a str must raise TypeError",
    )
