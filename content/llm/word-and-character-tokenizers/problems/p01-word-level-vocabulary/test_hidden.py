"""Hidden tests: a naive count-and-sort oracle, round trips, and the edge families.

Every failure message names the kind of mistake and carries no hidden input, id list or answer.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, NoReturn

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import WordTokenizer

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
UNK_ID = 1
ORACLE_WORDS = ["a", "A", "b", "cat", "Cat", "dog", "x.", "é", "<pad>", "<unk>"]
SEPARATORS = [" ", "  ", "\t", "\n"]
ROUND_TRIP_PIECES = ["cat", "Cat", "dog", "é", "x.y", "\t", "  ", "\n", " "]


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


def _naive_tokens(corpus: list[str], min_count: int) -> list[str]:
    """Independent oracle: count every word, keep the frequent ones, then sort on (-count, word)."""
    seen: dict[str, int] = {}
    for document in corpus:
        for piece in document.split():
            if piece not in SPECIALS:
                seen[piece] = seen.get(piece, 0) + 1
    rows = [(-count, word) for word, count in seen.items() if count >= min_count]
    rows.sort()
    return [*SPECIALS, *(word for _, word in rows)]


def _random_corpus(rng: np.random.Generator) -> list[str]:
    documents: list[str] = []
    for _ in range(int(rng.integers(0, 8))):
        pieces: list[str] = []
        for _ in range(int(rng.integers(0, 10))):
            pieces.append(ORACLE_WORDS[int(rng.integers(len(ORACLE_WORDS)))])
            pieces.append(SEPARATORS[int(rng.integers(len(SEPARATORS)))])
        documents.append("".join(pieces))
    return documents


def _build(corpus: list[str], min_count: int = 1) -> tuple[WordTokenizer | None, str | None]:
    try:
        return WordTokenizer(corpus, min_count), None
    except Exception as exc:
        return None, (
            f"your code raised {type(exc).__name__} on a valid corpus: check the counting loop "
            "and the cutoff"
        )


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_vocabulary_matches_the_naive_count_and_sort(seed: int) -> None:
    rng = np.random.default_rng(seed)
    corpus = _random_corpus(rng)
    min_count = int(rng.integers(1, 4))
    tok, problem = _build(corpus, min_count)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    expected = _naive_tokens(corpus, min_count)
    if tok.tokens != expected:
        _fail(
            "vocabulary order or membership differs: check the counts, the min_count cutoff, the "
            "descending count order and the alphabetical tie-break"
        )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_encoding_matches_the_naive_ids(seed: int) -> None:
    rng = np.random.default_rng(seed)
    corpus = _random_corpus(rng)
    min_count = int(rng.integers(1, 4))
    tok, problem = _build(corpus, min_count)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    # The specials are reserved: text that spells one of them is out of vocabulary, so only the
    # learned words (ids from 4 on) are looked up.
    vocab = {word: i for i, word in enumerate(_naive_tokens(corpus, min_count)) if i >= 4}
    text = " ".join(ORACLE_WORDS[int(rng.integers(len(ORACLE_WORDS)))] for _ in range(6))
    expected = [vocab.get(piece, UNK_ID) for piece in text.split()]
    try:
        got = tok.encode(text)
    except Exception as exc:
        error: str | None = type(exc).__name__
    else:
        error = None
    if error is not None:
        _fail(f"your encode raised {error} on a valid text: check the lookup")
    if got != expected:
        _fail(
            "encoded ids differ from the naive lookup: a word outside the vocabulary must get the "
            "<unk> id (1), and every word must be found with the same case and punctuation"
        )


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(pieces=st.lists(st.sampled_from(ROUND_TRIP_PIECES), max_size=20))
def test_round_trip_keeps_every_word_when_it_is_in_the_vocabulary(pieces: list[str]) -> None:
    text = "".join(pieces)
    tok, problem = _build([text], 1)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    try:
        got = tok.decode(tok.encode(text))
    except Exception as exc:
        error: str | None = type(exc).__name__
    else:
        error = None
    if error is not None:
        _fail(f"round trip raised {error}: check decode and the id lookup")
    if got != " ".join(text.split()):
        _fail(
            "round trip changed the words: decode must join the strings with single spaces, and "
            "encode must keep case and punctuation"
        )


def test_equal_counts_are_ordered_by_word_not_by_first_appearance() -> None:
    forward, _ = _build(["zeta alpha", "alpha zeta"])
    backward, _ = _build(["alpha zeta", "zeta alpha"])
    if forward is None or backward is None:
        _fail("the tokenizer could not be built on a valid corpus")
    if forward.tokens != [*SPECIALS, "alpha", "zeta"]:
        _fail("equal counts must be ordered alphabetically by the word, not by first appearance")
    if backward.tokens != forward.tokens:
        _fail("the vocabulary depends on corpus order: the id order must be deterministic")


def test_descending_count_comes_before_alphabetical_order() -> None:
    tok, problem = _build(["b b a", "c"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != [*SPECIALS, "b", "a", "c"]:
        _fail("a word with a higher count must get the smaller id, even when it sorts later")


def test_min_count_keeps_words_at_exactly_the_threshold() -> None:
    tok, problem = _build(["x x y", "y z"], 2)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != [*SPECIALS, "x", "y"]:
        _fail("min_count is inclusive: a word with count equal to min_count must be kept")


def test_counts_are_occurrences_not_document_frequencies() -> None:
    tok, problem = _build(["a a a", "b", "b"], 2)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != [*SPECIALS, "a", "b"]:
        _fail(
            "counts are occurrences: a word repeated inside one document must count each time, "
            "so it is not counted once per document"
        )


def test_case_and_punctuation_are_part_of_the_word() -> None:
    tok, problem = _build(["Cat cat cat."])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    # All three have count 1, so they are ordered by the word: "Cat" < "cat" < "cat.".
    if tok.tokens != [*SPECIALS, "Cat", "cat", "cat."]:
        _fail("case and punctuation must be kept: 'Cat', 'cat' and 'cat.' are three words")
    if tok.encode("CAT") != [UNK_ID]:
        _fail("an upper-case variant that is not in the vocabulary must encode to <unk>")


def test_tabs_newlines_and_repeated_spaces_separate_words() -> None:
    tok, problem = _build(["a\tb\n\nc   d", "  e  "])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != [*SPECIALS, "a", "b", "c", "d", "e"]:
        _fail("words are separated by any whitespace: split on runs, not on one space")
    if tok.encode("\n d \t") != [tok.tokens.index("d")]:
        _fail("leading and trailing whitespace must not create empty words")


def test_reserved_special_strings_in_the_corpus_do_not_take_ids() -> None:
    tok, problem = _build(["<pad> a <unk> <bos>"], 1)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != [*SPECIALS, "a"]:
        _fail("special token strings are reserved: a corpus word equal to one must not be added")
    if tok.encode("<pad>") != [UNK_ID]:
        _fail("a text word equal to a special token string is out of vocabulary and must be <unk>")
    if tok.decode([0, 1]) != "<pad> <unk>":
        _fail("the special ids keep their fixed order: 0 is <pad> and 1 is <unk>")


def test_empty_and_whitespace_only_text_give_no_ids() -> None:
    tok, problem = _build(["a b"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.encode("") != []:
        _fail("empty text must encode to an empty list, not to a special or unknown id")
    if tok.encode(" \t\n ") != []:
        _fail("whitespace-only text must encode to an empty list")
    if tok.decode([]) != "":
        _fail("decoding an empty id list must give the empty string")


def test_min_count_above_every_count_gives_only_the_specials() -> None:
    tok, problem = _build(["a b a"], 10**18)
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != SPECIALS:
        _fail("when no word reaches min_count, the vocabulary is the four special tokens only")


def test_ids_are_python_ints() -> None:
    tok, problem = _build(["a b"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    ids = tok.encode("a zzz b")
    for i in ids:
        if type(i) is not int:
            _fail(f"encode must return Python ints, got {type(i).__name__}: check lookups")


def test_large_vocabulary_has_dense_ids_in_sorted_order() -> None:
    words = [f"w{i}" for i in range(5000)]
    tok, problem = _build([" ".join(words)])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.tokens != SPECIALS + sorted(words):
        _fail("with equal counts, the ids must follow the alphabetical order of all 5000 words")
    probe = words[1234]
    if tok.encode(probe) != [tok.tokens.index(probe)]:
        _fail("a word in a large vocabulary must encode to its own id")


def test_corpus_and_text_are_not_mutated() -> None:
    corpus = ["b a", "c b"]
    before = copy.deepcopy(corpus)
    _build(corpus, 1)
    if corpus != before:
        _fail("the corpus was changed: do not sort, pop or edit the list or its strings")


def test_bare_string_corpus_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: WordTokenizer("a b c"),  # type: ignore[arg-type]  # invalid corpus
        "a single str as corpus must raise TypeError: it would be iterated character by character",
    )


def test_non_string_corpus_entry_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: WordTokenizer(["a", 2]),  # type: ignore[list-item]  # invalid entry
        "a non-str corpus entry must raise TypeError",
    )


def test_bool_min_count_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: WordTokenizer(["a"], True),  # invalid min_count
        "a bool min_count must raise TypeError: check type(min_count) is int",
    )


def test_float_min_count_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: WordTokenizer(["a"], 1.0),  # type: ignore[arg-type]  # invalid min_count
        "a float min_count must raise TypeError",
    )


def test_zero_and_negative_min_count_raise_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: WordTokenizer(["a"], 0),
        "min_count below 1 must raise ValueError",
    )
    _expect_error(
        ValueError,
        lambda: WordTokenizer(["a"], -3),
        "a negative min_count must raise ValueError",
    )


def test_non_string_text_raises_type_error_for_bytes() -> None:
    tok, problem = _build(["a"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    _expect_error(
        TypeError,
        lambda: tok.encode(b"a"),  # type: ignore[arg-type]  # invalid text
        "encode of bytes must raise TypeError: text must be a str",
    )


def test_decode_out_of_range_id_raises_value_error() -> None:
    tok, problem = _build(["a b"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    _expect_error(
        ValueError,
        lambda: tok.decode([len(tok.tokens)]),
        "an id past the end of the vocabulary must raise ValueError",
    )
    _expect_error(
        ValueError,
        lambda: tok.decode([-1]),
        "a negative id must raise ValueError: ids are indexes from 0",
    )


def test_decode_bool_id_raises_type_error() -> None:
    tok, problem = _build(["a"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    _expect_error(
        TypeError,
        lambda: tok.decode([True]),  # invalid id
        "a bool id must raise TypeError: check type(i) is int, since bool is a subclass of int",
    )


def test_decode_unk_gives_the_unk_string() -> None:
    tok, problem = _build(["a"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    if tok.decode([UNK_ID]) != "<unk>":
        _fail("decoding the <unk> id must give the string <unk>")


def test_decode_rejects_ids_that_are_not_a_list() -> None:
    tok, problem = _build(["a"])
    if problem is not None or tok is None:
        _fail(problem or "the tokenizer could not be built")
    _expect_error(
        TypeError,
        lambda: tok.decode((UNK_ID,)),  # type: ignore[arg-type]  # invalid ids
        "decode must reject ids that are a tuple: ids must be a list",
    )


def test_default_min_count_keeps_words_seen_once() -> None:
    tok = WordTokenizer(["a b", "a"])  # default min_count, not passed explicitly
    if tok.tokens != [*SPECIALS, "a", "b"]:
        _fail("the default min_count must be 1: a word that appears once must be kept")
