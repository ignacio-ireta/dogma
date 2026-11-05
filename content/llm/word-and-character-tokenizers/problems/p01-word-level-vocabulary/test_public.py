"""Public tests: the examples from problem.md and the basic edge cases.

The hidden tests check the tie-break, the cutoff, the reserved specials and the type rules.
"""

from __future__ import annotations

import pytest
from submission import WordTokenizer

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def test_vocabulary_order_is_specials_then_count_then_word() -> None:
    tok = WordTokenizer(["the cat sat", "the cat ran", "a dog sat"], min_count=2)
    assert tok.tokens == [*SPECIALS, "cat", "sat", "the"], (
        "vocabulary order is wrong: the specials come first, then words by descending count, "
        "and words with equal counts are ordered alphabetically"
    )
    assert tok.encode("the dog sat") == [6, 1, 5], (
        "encode gave the wrong ids: a word below min_count must get the <unk> id (1)"
    )
    assert tok.decode([6, 1, 5]) == "the <unk> sat", (
        "decode must join the token strings with single spaces; <unk> decodes to the string <unk>"
    )


def test_unknown_word_and_case_change_both_become_unk() -> None:
    tok = WordTokenizer(["the cat sat", "the cat ran", "a dog sat"], min_count=2)
    assert tok.encode("The\tcat  ") == [1, 4], (
        "'The' is not 'the' (case is kept), and tabs or repeated spaces separate words"
    )


def test_empty_corpus_has_only_the_special_tokens() -> None:
    tok = WordTokenizer([])
    assert tok.tokens == SPECIALS, "an empty corpus must still give the four special tokens"
    assert tok.encode("") == [], "empty text must encode to an empty list"


def test_non_string_text_raises_type_error() -> None:
    tok = WordTokenizer(["a"])
    try:
        tok.encode(3)  # type: ignore[arg-type]  # the text must be a str
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(
            f"encode of a non-str must raise TypeError, instead it raised {type(exc).__name__}"
        )
    pytest.fail("encode accepted a non-str text: check that text is a str and raise TypeError")
