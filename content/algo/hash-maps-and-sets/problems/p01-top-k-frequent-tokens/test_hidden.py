"""Hidden tests: an oracle comparison with collections.Counter, plus edge cases.

Each failure message names the kind of mistake it points to.
"""

from __future__ import annotations

import collections

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import top_k_frequent  # type: ignore[import-not-found]

VOCAB = ["a", "b", "c", "B", "ab", "", "zz"]


def _oracle(tokens: list[str], k: int) -> list[tuple[str, int]]:
    """Independent reference: Counter for the counts, then an explicit sort."""
    counts = collections.Counter(tokens)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[: max(k, 0)]


@settings(max_examples=150, deadline=None, report_multiple_bugs=False)
@given(
    tokens=st.lists(st.sampled_from(VOCAB), max_size=60),
    k=st.integers(min_value=-3, max_value=12),
)
def test_matches_counter_oracle_on_random_token_lists(tokens: list[str], k: int) -> None:
    assert top_k_frequent(tokens, k) == _oracle(tokens, k), (
        "result differs from the Counter reference: check the counts, the (count descending, "
        "token ascending) order, and the k <= 0 and k larger-than-vocabulary cases"
    )


def test_matches_counter_oracle_on_a_large_seeded_stream() -> None:
    rng = np.random.default_rng(0)
    vocab = [f"tok{i}" for i in range(400)]
    tokens = [vocab[int(i)] for i in rng.integers(0, len(vocab), size=40_000)]
    assert top_k_frequent(tokens, 25) == _oracle(tokens, 25), (
        "large stream: the counts or the ranking are wrong. Check that every token is counted "
        "and that ties are ordered by token"
    )


def test_zero_and_negative_k_return_an_empty_list_for_any_input() -> None:
    for k in (0, -1, -100):
        assert top_k_frequent(["a", "a", "b"], k) == [], (
            "k <= 0 should return an empty list; a negative slice would return wrong rows"
        )


def test_empty_token_list_returns_an_empty_list_for_any_k() -> None:
    for k in (0, 1, 5):
        assert top_k_frequent([], k) == [], "no tokens should give [] for any k"


def test_tokens_are_compared_case_sensitively() -> None:
    assert top_k_frequent(["Apple", "apple", "apple"], 2) == [("apple", 2), ("Apple", 1)], (
        "Apple and apple are different tokens: do not lower-case or strip the input"
    )


def test_empty_string_is_an_ordinary_token_that_gets_counted() -> None:
    assert top_k_frequent(["", "", "x"], 2) == [("", 2), ("x", 1)], (
        "the empty string is a valid token: count it, do not skip it"
    )


def test_ties_are_sorted_by_string_order_not_by_first_appearance() -> None:
    assert top_k_frequent(["zeta", "alpha", "mid"], 3) == [
        ("alpha", 1),
        ("mid", 1),
        ("zeta", 1),
    ], "equal counts should be ordered by token, not by the order they first appeared"
    assert top_k_frequent(["b", "B", "a"], 3) == [("B", 1), ("a", 1), ("b", 1)], (
        "tie order is Python string order: uppercase 'B' sorts before lowercase 'a'"
    )


def test_result_is_a_list_of_str_int_tuples() -> None:
    result = top_k_frequent(["a", "b", "a"], 2)
    assert isinstance(result, list), f"return a list, got {type(result).__name__}"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result), (
        "each item must be a (token, count) tuple, not a list or a Counter entry"
    )
    assert all(isinstance(item[0], str) and isinstance(item[1], int) for item in result), (
        "each tuple must hold a str token and an int count"
    )


def test_input_list_is_not_mutated() -> None:
    tokens = ["b", "a", "b"]
    snapshot = list(tokens)
    top_k_frequent(tokens, 2)
    assert tokens == snapshot, (
        "the input list was modified: do not sort or change the tokens in place"
    )
