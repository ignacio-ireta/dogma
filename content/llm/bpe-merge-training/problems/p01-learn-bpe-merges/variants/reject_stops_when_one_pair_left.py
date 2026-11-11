from __future__ import annotations

# The end-of-word marker. It is one symbol that follows the last character of every word, so a
# merge can join it to the final character (for example "r" and "</w>" become "r</w>").
END = "</w>"


def _merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every occurrence of pair in symbols, scanning left to right without overlaps.

    In "aaa" with pair ("a", "a") the first two symbols merge, and the third stays as it is.
    """
    left, right = pair
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
            out.append(left + right)  # the new symbol is the concatenation of the two parts
            i += 2  # both symbols are consumed, so the scan resumes after them
        else:
            out.append(symbols[i])
            i += 1
    return out


def _validate(word_counts: dict[str, int], num_merges: int) -> None:
    """Reject bad input before any work happens, so the learner sees the rule that was broken."""

    if type(word_counts) is not dict:
        raise TypeError("word_counts must be a dict of str to int")
    if type(num_merges) is not int:
        raise TypeError(f"num_merges must be a Python int, got {type(num_merges).__name__}")
    if num_merges < 0:
        raise ValueError("num_merges must be zero or more")
    for word, count in word_counts.items():
        if type(word) is not str:
            raise TypeError(f"every word must be a str, got {type(word).__name__}")
        if type(count) is not int:
            raise TypeError("every count must be a Python int (bool is rejected)")
        if word == "":
            raise ValueError("a word must not be empty")
        if any(ch.isspace() for ch in word):
            raise ValueError("a word must not contain whitespace: split the text first")
        if END in word:
            raise ValueError("a word must not contain the end-of-word marker text")
        if count < 1:
            raise ValueError("every count must be at least 1")


def train_bpe_merges(word_counts: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Wrong on purpose: learn up to num_merges merges from a table of word counts, in the order they are learned.

    Each step counts every adjacent symbol pair in every word, weighted by the word's count.
    An overlapping occurrence counts, so "aaa" contributes two (a, a) pairs. The pair with the
    highest count is merged. Ties go to the smallest pair under tuple order, so ("a", "b") comes
    before ("a", "c"), and ("a", "</w>") comes before ("a", "b").
    """  # noqa: E501
    _validate(word_counts, num_merges)

    # Step 1: each word is a list of symbols, its characters followed by the marker. Merges are
    # learned on this word table, not on the raw text, so no pair can cross a word boundary.
    # Each entry pairs the word's count (its weight) with its current symbols.
    entries: list[tuple[int, list[str]]] = [
        (count, [*word, END]) for word, count in word_counts.items()
    ]

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        # Step 2: count the adjacent pairs. range(len - 1) visits every position, including
        # repeated pairs that overlap, which is what the "aaa" rule above requires.
        counts: dict[tuple[str, str], int] = {}
        for weight, symbols in entries:
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                counts[pair] = counts.get(pair, 0) + weight
        if len(counts) < 2:
            break  # every word is a single symbol, so no merge is left to learn

        # Step 3: the best pair has the largest count. Negating the count and comparing the pairs
        # as tuples gives "largest count, then smallest pair" in one min().
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)

        # Step 4: apply the merge to every word, so the next count sees the new symbol.
        entries = [(weight, _merge_pair(symbols, best)) for weight, symbols in entries]
    return merges
