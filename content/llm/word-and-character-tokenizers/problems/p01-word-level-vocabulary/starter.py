from __future__ import annotations

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


class WordTokenizer:
    """A word-level tokenizer whose vocabulary is learned from a corpus.

    A word is a maximal run of non-whitespace characters, as given by str.split() with no
    argument. Case and punctuation are kept. Special token strings are reserved.

    Attributes
    ----------
    tokens : list of str. tokens[i] is the string with id i. The first four entries are
        <pad>, <unk>, <bos>, <eos>, in that order, and the learned words follow.
    """

    def __init__(self, corpus: list[str], min_count: int = 1) -> None:
        """Learn the vocabulary from corpus.

        corpus : a list of str. Must not be mutated.
        min_count : a Python int, at least 1. A word is kept when it occurs this often or more.

        Raises TypeError for a non-list corpus, a non-str entry, or a min_count that is not a
        Python int (bool is rejected). Raises ValueError when min_count is below 1.
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Return the id of each word of text. Words outside the vocabulary get the UNK id.

        Raises TypeError when text is not a str.
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Return the token strings for ids, joined with single spaces.

        Raises TypeError for a non-list or for an id that is not a Python int (bool is rejected).
        Raises ValueError for an id outside 0 .. len(tokens) - 1.
        """
        raise NotImplementedError
