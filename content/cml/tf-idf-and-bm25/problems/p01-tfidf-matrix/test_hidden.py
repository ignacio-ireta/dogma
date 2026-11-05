"""Hidden tests: scikit-learn oracles, the tokenizer, idf and normalisation rules, and errors.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.feature_extraction.text import TfidfVectorizer
from submission import tfidf_matrix

LARGE_DOCS = 3_000
LARGE_TIME_LIMIT_S = 5.0
WORDS = ["cat", "Cat", "dog", "DOG", "a", "I", "x", "42", "7", "the", "sat", "mat", "café", "naïve"]
SEPARATORS = [" ", " ", " ", ", ", ". ", "-", "'", "  "]
MATRIX_MISTAKE = (
    "the matrix differs from the reference: tokens are lowercased runs of two or more word "
    "characters, idf is ln((1 + n) / (1 + df)) + 1, and each row is divided by its l2 norm"
)


def _fail(why: str) -> None:
    raise AssertionError(why)


def _expect_value_error(call: Callable[[], object], mistake: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        _fail(f"{mistake} (raised {type(exc).__name__} instead)")
    _fail(mistake)


def _solve(docs: Any, **kwargs: Any) -> tuple[np.ndarray, list[str]]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        matrix, vocab = tfidf_matrix(docs, **kwargs)
    except Exception as exc:
        _fail(
            f"valid input raised {type(exc).__name__}: check the tokens, the idf and the vocabulary"
        )
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64:
        _fail("return the matrix as a float64 NumPy array")
    if not isinstance(vocab, list) or not all(isinstance(t, str) for t in vocab):
        _fail("return the vocabulary as a list of str")
    return matrix, vocab


def _random_docs(rng: np.random.Generator, n_docs: int) -> list[str]:
    docs = []
    for _ in range(n_docs):
        length = int(rng.integers(0, 9))
        pieces = []
        for _ in range(length):
            pieces.append(WORDS[int(rng.integers(0, len(WORDS)))])
            pieces.append(SEPARATORS[int(rng.integers(0, len(SEPARATORS)))])
        docs.append("".join(pieces))
    return docs


def _reference(
    docs: list[str], vocabulary: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """scikit-learn's default TfidfVectorizer, the oracle for the matrix and the vocabulary."""
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    matrix = vectorizer.fit_transform(docs).toarray()
    return matrix, list(vectorizer.get_feature_names_out())


@pytest.mark.parametrize("seed", range(12), ids=[f"corpus-{s}" for s in range(12)])
def test_matches_scikit_learn_on_random_corpora(seed: int) -> None:
    rng = np.random.default_rng(seed)
    docs = _random_docs(rng, int(rng.integers(2, 12)))
    if not any(_has_token(doc) for doc in docs):
        docs.append("cat dog")  # keep the vocabulary non-empty; the empty case is tested below
    want_matrix, want_vocab = _reference(docs)
    got_matrix, got_vocab = _solve(docs)
    assert got_vocab == want_vocab, (
        "the vocabulary differs: lowercase first, keep runs of two or more word characters, and "
        "sort the terms"
    )
    assert got_matrix.shape == want_matrix.shape, (
        "matrix shape: one row per document, one column per term"
    )
    assert np.allclose(got_matrix, want_matrix, rtol=1e-9, atol=1e-12), MATRIX_MISTAKE


@pytest.mark.parametrize("seed", range(6), ids=[f"fixed-{s}" for s in range(6)])
def test_matches_scikit_learn_with_a_fixed_vocabulary(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    docs = _random_docs(rng, int(rng.integers(2, 10)))
    if not any(_has_token(doc) for doc in docs):
        docs.append("cat dog")
    vocabulary = ["sat", "cat", "zebra", "dog", "42", "mat"][: int(rng.integers(1, 7))]
    want_matrix, want_vocab = _reference(docs, vocabulary)
    got_matrix, got_vocab = _solve(docs, vocabulary=vocabulary)
    assert got_vocab == want_vocab == vocabulary, (
        "a given vocabulary fixes the columns in its own order; other tokens are dropped"
    )
    assert np.allclose(got_matrix, want_matrix, rtol=1e-9, atol=1e-12), MATRIX_MISTAKE


def _has_token(doc: str) -> bool:
    return any(
        len(word) >= 2 for word in "".join(c if c.isalnum() else " " for c in doc.lower()).split()
    )


def test_vocabulary_is_sorted_when_not_given() -> None:
    _, vocab = _solve(["zebra Apple", "42 apple"])
    assert vocab == ["42", "apple", "zebra"], (
        "the default vocabulary is every token in ascending string order, not first-seen order"
    )


def test_single_characters_are_not_terms_but_digits_of_two_characters_are() -> None:
    _, vocab = _solve(["I am a 7 cat, x 42"])
    assert vocab == ["42", "am", "cat"], (
        "tokenisation must drop single-character words and preserve word-charac"
        "ter tokens of the stated minimum length"
    )


def test_term_in_every_document_has_idf_one() -> None:
    matrix, vocab = _solve(["cat dog", "cat"])
    idf_dog = math.log(3 / 2) + 1
    expected = np.array([[1.0, idf_dog], [1.0, 0.0]]) / np.array(
        [[math.hypot(1.0, idf_dog)], [1.0]]
    )
    assert vocab == ["cat", "dog"], (
        "the default vocabulary must contain the surviving tokens in sorted order"
    )
    assert np.allclose(matrix, expected, rtol=1e-12), (
        "a term present in every document must have the stated baseline idf"
    )


def test_rows_have_unit_length_and_empty_documents_stay_zero() -> None:
    matrix, _ = _solve(["cat dog", "", "a 7 I", "dog dog cat"])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms[[0, 3]], 1.0, rtol=1e-12), (
        "each non-empty row must be divided by its l2 norm, so its length is 1"
    )
    assert np.all(matrix[1:3] == 0.0), (
        "a document with no token of two characters has no weight: its row must stay zeros, not NaN"
    )
    assert np.all(np.isfinite(matrix)), "no NaN or infinity: a zero norm must not be divided"


def test_counts_are_raw_not_logarithmic() -> None:
    matrix, vocab = _solve(["cat cat cat dog", "dog"])
    idf_cat = math.log(3 / 2) + 1  # cat is in one of the two documents
    idf_dog = math.log(3 / 3) + 1  # dog is in both documents, so it has idf 1
    raw = np.array([3 * idf_cat, 1 * idf_dog])
    expected_row0 = raw / np.linalg.norm(raw)
    assert vocab == ["cat", "dog"], (
        "the default vocabulary must contain the surviving tokens in sorted order"
    )
    assert np.allclose(matrix[0], expected_row0, rtol=1e-12), (
        "term frequency must use raw counts, without logarithmic scaling"
    )


def test_empty_vocabulary_raises_value_error() -> None:
    _expect_value_error(
        lambda: tfidf_matrix(["a I 7", ""]),
        "a corpus with no token of two or more characters was accepted: raise ValueError",
    )
    _expect_value_error(
        lambda: tfidf_matrix(["cat"], vocabulary=[]),
        "an empty vocabulary was accepted: raise ValueError when there are no terms",
    )


def test_repeated_vocabulary_term_raises_value_error() -> None:
    _expect_value_error(
        lambda: tfidf_matrix(["cat dog"], vocabulary=["cat", "cat"]),
        "a repeated vocabulary term was accepted: raise ValueError",
    )


def test_inputs_are_not_mutated() -> None:
    docs = ["cat sat", "cat cat", "dog"]
    vocab = ["sat", "cat"]
    before_docs, before_vocab = list(docs), list(vocab)
    _solve(docs, vocabulary=vocab)
    assert docs == before_docs, "docs was changed: tokenize without editing the caller's list"
    assert vocab == before_vocab, "vocabulary was reordered or changed: copy it before using it"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.sampled_from(WORDS), min_size=2, max_size=8))
def test_every_non_empty_row_is_unit_length(words: list[str]) -> None:
    docs = [" ".join(words[:4]) or "cat", " ".join(words[4:]) or "dog", "cat dog sat"]
    matrix, _ = _solve(docs)
    norms = np.linalg.norm(matrix, axis=1)
    for norm in norms:
        assert norm == pytest.approx(0.0, abs=1e-12) or norm == pytest.approx(1.0, rel=1e-12), (
            "each row must have l2 norm 0 (no weight) or exactly 1: check the normalisation"
        )


def test_large_corpus_finishes_quickly_and_matches_scikit_learn() -> None:
    rng = np.random.default_rng(5)
    docs = _random_docs(rng, LARGE_DOCS)
    docs.append("cat dog")
    want_matrix, want_vocab = _reference(docs)
    start = time.perf_counter()
    got_matrix, got_vocab = _solve(docs)
    elapsed = time.perf_counter() - start
    assert elapsed < LARGE_TIME_LIMIT_S, (
        "a large corpus is too slow: count each document once and fill the matrix row by row"
    )
    assert got_vocab == want_vocab, "the vocabulary differs on a large corpus: check the tokenizer"
    assert np.allclose(got_matrix, want_matrix, rtol=1e-9, atol=1e-12), MATRIX_MISTAKE


def test_empty_corpus_raises_even_with_a_fixed_vocabulary():
    _expect_value_error(
        lambda: tfidf_matrix([], vocabulary=["cat", "dog"]),
        "an empty corpus must raise ValueError even when a vocabulary is given",
    )
