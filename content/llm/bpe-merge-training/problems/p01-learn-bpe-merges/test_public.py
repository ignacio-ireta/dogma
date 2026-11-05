"""Public tests: the worked examples from problem.md, the overlap rule and the empty table.

The hidden tests check the tie-break against a naive trainer, the weights and the type rules.
"""

from __future__ import annotations

from submission import train_bpe_merges


def test_worked_example_learns_three_merges_with_a_tie_break() -> None:
    got = train_bpe_merges({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 3)
    assert got == [("e", "s"), ("es", "t"), ("est", "</w>")], (
        "the first three merges are wrong: (e, s), (s, t) and (t, </w>) tie at 9, and the tie goes "
        "to the smallest pair; each later count must use the words after the earlier merges"
    )


def test_overlapping_pairs_are_counted_at_every_position() -> None:
    assert train_bpe_merges({"aaa": 1, "bb": 1}, 1) == [("a", "a")], (
        "in 'aaa' the pair (a, a) occurs at two positions, so it counts 2 and wins; counting only "
        "non-overlapping occurrences would tie it with (a, </w>)"
    )


def test_empty_table_and_zero_merges_return_an_empty_list() -> None:
    assert train_bpe_merges({}, 5) == [], "an empty table has no pairs, so there are no merges"
    assert train_bpe_merges({"low": 1}, 0) == [], "zero merges requested must return an empty list"
