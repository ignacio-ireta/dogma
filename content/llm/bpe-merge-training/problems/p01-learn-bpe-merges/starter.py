from __future__ import annotations


def train_bpe_merges(word_counts: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Learn up to num_merges byte-pair-encoding merges from a table of word counts.

    Each word starts as its characters followed by the end-of-word marker "</w>", a separate
    symbol. Each step counts the adjacent symbol pairs, weighted by word count, and merges the
    most frequent pair in every word. An overlapping occurrence counts. A tie goes to the
    smallest pair under tuple order. Stops early when no pair is left.

    Parameters
    ----------
    word_counts : dict mapping each word (non-empty str, no whitespace, not containing "</w>")
        to its count (Python int, at least 1). Must not be mutated.
    num_merges : Python int, at least 0.

    Returns
    -------
    list of (left, right) tuples, in the order the merges were learned. The merged symbol is
    left + right. The list has at most num_merges items.

    Raises
    ------
    TypeError
        When word_counts is not a dict, a word is not a str, a count or num_merges is not a
        Python int (bool is rejected).
    ValueError
        When a word is empty, contains whitespace or the text "</w>", a count is below 1, or
        num_merges is negative.
    """
    raise NotImplementedError
