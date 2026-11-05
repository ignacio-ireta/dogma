from __future__ import annotations


class BPETokenizer:
    """Encode words with a ranked list of BPE merges, and decode token lists back to words.

    The rank of a merge is its index in the list. Lower rank is applied first.
    Each word is encoded as its characters followed by the end-of-word marker "</w>".
    """

    def __init__(self, merges: list[tuple[str, str]]) -> None:
        """Store the merges and their ranks.

        Raises TypeError when merges is not a list of (left, right) tuples of two str.
        Raises ValueError when a symbol is empty or the same pair appears twice.
        """
        raise NotImplementedError

    def encode(self, word: str) -> list[str]:
        """Return the symbols of word after applying the merges by rank.

        Raises TypeError when word is not a str. Raises ValueError when word contains whitespace
        or the text "</w>".
        """
        raise NotImplementedError

    def decode(self, tokens: list[str]) -> str:
        """Return the word spelled by tokens: concatenate them and remove the trailing "</w>".

        Raises TypeError when tokens is not a list of str. Raises ValueError when the joined text
        does not end with "</w>", or when "</w>" appears anywhere else.
        """
        raise NotImplementedError
