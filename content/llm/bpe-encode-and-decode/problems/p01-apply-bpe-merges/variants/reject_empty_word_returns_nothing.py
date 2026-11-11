from __future__ import annotations

from itertools import pairwise

# The end-of-word marker, the same symbol that the trainer adds after every word.
END = "</w>"


class BPETokenizer:
    """Wrong on purpose: the empty word encodes to an empty list. Encode words with a ranked merge list, and decode token lists back to words.

    The rank of a merge is its position in the list. Lower rank means the merge was learned earlier,
    and it is applied first whenever it is available.
    """  # noqa: E501

    def __init__(self, merges: list[tuple[str, str]]) -> None:
        # Validate the list, then store each pair's rank in a dict. A pair appearing twice has two
        # ranks, which makes the order ambiguous, so it is rejected.
        if type(merges) is not list:
            raise TypeError("merges must be a list of (left, right) tuples")
        ranks: dict[tuple[str, str], int] = {}
        for index, item in enumerate(merges):
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("each merge must be a (left, right) tuple of two str")
            left, right = item
            if type(left) is not str or type(right) is not str:
                raise TypeError("each merge must be a (left, right) tuple of two str")
            if left == "" or right == "":
                raise ValueError("a merge symbol must not be empty")
            if item in ranks:
                raise ValueError("a merge appears twice: each pair needs exactly one rank")
            ranks[item] = index
        self._ranks = ranks

    def encode(self, word: str) -> list[str]:
        """Return the symbols of word after all applicable merges, in order.

        The word starts as its characters followed by the end-of-word marker.
        """
        if type(word) is not str:
            raise TypeError(f"word must be a str, got {type(word).__name__}")
        if any(ch.isspace() for ch in word):
            raise ValueError("a word must not contain whitespace: split the text first")
        if END in word:
            raise ValueError("a word must not contain the end-of-word marker text")

        if word == "":
            return []
        symbols = [*word, END]
        while True:
            # Step 1: list the adjacent pairs that are merges, with their ranks. The lowest rank
            # is the merge learned first, and it is applied before any higher-rank merge.
            candidates = [
                (self._ranks[pair], pair) for pair in pairwise(symbols) if pair in self._ranks
            ]
            if not candidates:
                break  # no adjacent pair is a merge any more, so the word is fully encoded
            _, pair = min(candidates)

            # Step 2: merge every occurrence of that pair, scanning left to right. After a match
            # both symbols are consumed, so "aaa" with (a, a) gives "aa" and "a", not "a" and "aa".
            left, right = pair
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    def decode(self, tokens: list[str]) -> str:
        """Return the word that tokens spell. Concatenate, then strip the trailing marker."""
        if type(tokens) is not list or any(type(t) is not str for t in tokens):
            raise TypeError("tokens must be a list of str")
        # Concatenation is the inverse of the merges: each merged symbol is the join of its parts.
        text = "".join(tokens)
        if not text.endswith(END):
            raise ValueError("the tokens must end with the end-of-word marker")
        word = text[: -len(END)]
        if END in word:
            raise ValueError("the end-of-word marker may appear only at the end")
        return word
