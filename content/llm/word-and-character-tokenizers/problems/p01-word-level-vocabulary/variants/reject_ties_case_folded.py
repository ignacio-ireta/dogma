from __future__ import annotations

# Reserved special tokens. Their ids are fixed by position: pad=0, unk=1, bos=2, eos=3.
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)
UNK_ID = SPECIALS.index(UNK)


class WordTokenizer:
    """Wrong on purpose: equal counts are ordered by the lowercase word.

    A word is a maximal run of non-whitespace characters, as given by str.split() with no
    argument. Case and punctuation are kept, so "Cat", "cat" and "cat." are three words.

    Attributes
    ----------
    tokens : list of str. tokens[i] is the string with id i. The first four entries are the
        special tokens in the order <pad>, <unk>, <bos>, <eos>.
    """

    def __init__(self, corpus: list[str], min_count: int = 1) -> None:
        # Validate first, so a bad argument is named before any counting happens.
        # type(...) is int rejects bool, which is a subclass of int.
        if type(min_count) is not int:
            raise TypeError(f"min_count must be a Python int, got {type(min_count).__name__}")
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        if type(corpus) is not list:
            raise TypeError("corpus must be a list of str, not a single string or another type")
        for document in corpus:
            if type(document) is not str:
                raise TypeError(f"every corpus entry must be a str, got {type(document).__name__}")

        # Step 1: count occurrences. Every appearance counts, so a word that appears three times
        # in one document has count 3. Special token strings are reserved, so they are skipped.
        counts: dict[str, int] = {}
        for document in corpus:
            for word in document.split():
                if word in SPECIALS:
                    continue
                counts[word] = counts.get(word, 0) + 1

        # Step 2: apply the cutoff. A word stays when its count is at least min_count.
        kept = [word for word in counts if counts[word] >= min_count]

        # Step 3: order the learned words by descending count. Ties go to the lexicographically
        # smaller word, so the id order does not depend on the order of the corpus.
        kept.sort(key=lambda word: (-counts[word], word.lower()))

        # Step 4: ids. The specials keep ids 0..3, and the learned words follow.
        self.tokens: list[str] = list(SPECIALS) + kept
        self._ids: dict[str, int] = {word: len(SPECIALS) + i for i, word in enumerate(kept)}

    def encode(self, text: str) -> list[int]:
        """Map each word of text to its id, and map any word outside the vocabulary to UNK."""
        if type(text) is not str:
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        # Specials are not in self._ids, so the text string "<pad>" maps to UNK like any unknown
        # word. Words are split with the same rule as the corpus, and case is not changed.
        return [self._ids.get(word, UNK_ID) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        """Map ids back to their token strings and join them with single spaces."""
        if type(ids) is not list:
            raise TypeError("ids must be a list of Python int")
        for i in ids:
            if type(i) is not int:
                raise TypeError(f"every id must be a Python int, got {type(i).__name__}")
            if not 0 <= i < len(self.tokens):
                raise ValueError(f"id {i} is outside the vocabulary of {len(self.tokens)} tokens")
        return " ".join(self.tokens[i] for i in ids)
