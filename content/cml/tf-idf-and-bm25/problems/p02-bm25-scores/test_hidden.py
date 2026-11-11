"""Hidden tests: a NumPy oracle, the properties of the formula, empty inputs, errors and speed.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import bm25_scores

VOCAB = ["cat", "dog", "sat", "mat", "bird", "fish", "the"]
LARGE_DOCS = 5_000
LARGE_TIME_LIMIT_S = 5.0
SCORE_MISTAKE = (
    "a score differs from the reference: idf = ln(1 + (N - n + 0.5) / (n + 0.5)), avgdl averages "
    "every document, and the term part is f * (k1 + 1) / (f + k1 * (1 - b + b * |D| / avgdl))"
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


def _solve(query: Any, docs: Any, **kwargs: Any) -> list[float]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        got = bm25_scores(query, docs, **kwargs)
    except Exception as exc:
        _fail(
            f"valid input raised {type(exc).__name__}: check the idf, the saturation, the average "
            "length and the empty documents"
        )
    if not isinstance(got, list) or len(got) != len(docs):
        _fail("return one score per document, as a list, in input order")
    if not all(isinstance(s, float) for s in got):
        _fail("return every score as a Python float")
    return got


def _oracle(query: list[str], docs: list[list[str]], k1: float, b: float) -> np.ndarray:
    """Independent computation: a term-frequency matrix and vectorised arithmetic."""
    n_docs = len(docs)
    terms = sorted({t for d in docs for t in d} | set(query))
    column = {t: j for j, t in enumerate(terms)}
    tf = np.zeros((n_docs, len(terms)))
    for i, doc in enumerate(docs):
        for token in doc:
            tf[i, column[token]] += 1
    lengths = np.array([len(d) for d in docs], dtype=float)
    avgdl = lengths.mean()
    df = (tf > 0).sum(axis=0)
    total = np.zeros(n_docs)
    for token in query:
        j = column[token]
        if df[j] == 0:
            continue
        idf = math.log(1 + (n_docs - df[j] + 0.5) / (df[j] + 0.5))
        f = tf[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = 1 - b + b * lengths / avgdl if avgdl > 0 else np.ones(n_docs)
            term = np.where(f > 0, idf * f * (k1 + 1) / (f + k1 * norm), 0.0)
        total += term
    return total


def _random_collection(rng: np.random.Generator) -> tuple[list[str], list[list[str]], float, float]:
    n_docs = int(rng.integers(1, 9))
    docs = [
        [VOCAB[int(rng.integers(0, len(VOCAB)))] for _ in range(int(rng.integers(0, 13)))]
        for _ in range(n_docs)
    ]
    query = [VOCAB[int(rng.integers(0, len(VOCAB)))] for _ in range(int(rng.integers(0, 5)))]
    k1 = float(rng.uniform(0.0, 3.0))
    b = float(rng.uniform(0.0, 1.0))
    return query, docs, k1, b


@pytest.mark.parametrize("seed", range(20), ids=[f"random-{s}" for s in range(20)])
def test_matches_the_numpy_reference_on_random_collections(seed: int) -> None:
    rng = np.random.default_rng(seed)
    query, docs, k1, b = _random_collection(rng)
    want = _oracle(query, docs, k1, b)  # computed before the learner runs
    got = _solve(query, docs, k1=k1, b=b)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12), SCORE_MISTAKE


def test_term_frequency_saturates_and_stays_below_idf_times_k1_plus_one() -> None:
    k1 = 1.2
    scores = []
    for f in range(1, 10):
        doc = ["cat"] * f + ["dog"] * (10 - f)  # fixed length 10, so only f changes
        scores.append(_solve(["cat"], [doc, ["sat"] * 10])[0])
    assert all(a < c for a, c in pairwise(scores)), (
        "the score must rise with the term count, though more slowly each time"
    )
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    assert max(scores) < idf * (k1 + 1), (
        "saturation: the term part never reaches idf * (k1 + 1), however large the count"
    )
    gains = np.diff(scores)
    assert all(a > c for a, c in pairwise(gains)), (
        "saturation means each extra occurrence adds less than the one before"
    )


def test_with_b_zero_document_length_does_not_matter() -> None:
    short = ["cat", "dog"]
    long = ["cat", "dog"] + ["sat"] * 9
    got = _solve(["cat"], [short, long, ["sat"]], b=0.0)
    assert math.isclose(got[0], got[1], rel_tol=1e-12), (
        "with b = 0 there is no length normalisation: equal counts must score equally"
    )


def test_with_b_one_a_longer_document_with_the_same_count_scores_lower() -> None:
    short = ["cat", "dog"]
    long = ["cat", "dog"] + ["sat"] * 6
    got = _solve(["cat"], [short, long], b=1.0)
    assert got[1] < got[0], (
        "with b = 1 a longer document than average is penalised: check the length term"
    )


def test_with_k1_zero_each_matching_document_scores_exactly_its_idf() -> None:
    docs = [["cat"] * 5, ["cat"], ["dog"], ["dog", "fish"]]
    got = _solve(["cat"], docs, k1=0.0)
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    assert math.isclose(got[0], idf, rel_tol=1e-12) and math.isclose(got[1], idf, rel_tol=1e-12), (
        "with k1 = 0 the term part is idf for any count of the term"
    )
    assert got[2] == 0.0 and got[3] == 0.0, "documents without the term must score 0.0"


def test_repeated_query_token_counts_twice() -> None:
    docs = [["cat", "dog"], ["dog", "sat"], ["cat"]]
    once = _solve(["cat"], docs)
    twice = _solve(["cat", "cat"], docs)
    assert all(math.isclose(t, 2 * o, rel_tol=1e-12) for t, o in zip(twice, once, strict=False)), (
        "each occurrence of a query token contributes its own term part: the query is a sum"
    )


def test_a_term_in_most_documents_still_adds_a_positive_score() -> None:
    docs = [["cat"], ["cat"], ["cat"], ["dog"]]
    got = _solve(["cat"], docs)
    assert all(s > 0.0 for s in got[:3]), (
        "a common matching term must add a positive score under the Lucene idf convention"
    )
    assert got[3] == 0.0, "the document without the term must score 0.0"


def test_average_length_includes_empty_documents() -> None:
    docs = [["cat"], [], []]
    got = _solve(["cat"], docs, k1=1.2, b=0.75)
    avgdl = 1 / 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * (1 + 1.2) / (1 + 1.2 * (1 - 0.75 + 0.75 * 1 / avgdl))
    assert math.isclose(got[0], expected, rel_tol=1e-12), (
        "avgdl is the mean over all documents, empty ones included, not over the non-empty ones"
    )


def test_collection_of_empty_documents_gives_zeros() -> None:
    got = _solve(["cat"], [[], [], []])
    assert got == [0.0, 0.0, 0.0], (
        "when every document is empty avgdl is 0: return 0.0 scores without dividing by zero"
    )


def test_reordering_the_documents_reorders_the_scores() -> None:
    rng = np.random.default_rng(11)
    query, docs, k1, b = _random_collection(rng)
    docs = [*docs, ["cat", "dog", "cat"]]
    order = list(range(len(docs)))[::-1]
    forward = _solve(query, docs, k1=k1, b=b)
    backward = _solve(query, [docs[i] for i in order], k1=k1, b=b)
    assert np.allclose(forward, backward[::-1], rtol=1e-12, atol=1e-12), (
        "the score of a document must not depend on where it sits in the collection"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.sampled_from(VOCAB), min_size=0, max_size=4),
    st.lists(st.lists(st.sampled_from(VOCAB), min_size=0, max_size=8), min_size=1, max_size=6),
)
def test_scores_are_never_negative(query: list[str], docs: list[list[str]]) -> None:
    got = _solve(query, docs)
    assert all(s >= 0.0 for s in got), (
        "BM25 scores with the Lucene idf are never negative: check the sign of the idf"
    )


def test_invalid_arguments_raise_value_error() -> None:
    _expect_value_error(
        lambda: bm25_scores(["cat"], []),
        "an empty collection was accepted: raise ValueError when there are no documents",
    )
    _expect_value_error(
        lambda: bm25_scores(["cat"], [["cat"]], k1=-0.5),
        "a negative k1 was accepted: raise ValueError",
    )
    _expect_value_error(
        lambda: bm25_scores(["cat"], [["cat"]], b=1.5),
        "a b above 1 was accepted: raise ValueError, b must lie in [0, 1]",
    )
    _expect_value_error(
        lambda: bm25_scores(["cat"], [["cat"]], b=-0.1),
        "a b below 0 was accepted: raise ValueError, b must lie in [0, 1]",
    )


def test_inputs_are_not_mutated() -> None:
    query = ["cat", "dog"]
    docs = [["cat", "sat"], ["dog", "dog"], []]
    before_query = list(query)
    before_docs = [list(d) for d in docs]
    _solve(query, docs)
    assert query == before_query, "query_tokens was changed: do not edit the caller's query"
    assert docs == before_docs, "a document was changed: count tokens without editing them"


def test_large_collection_finishes_quickly_and_matches_the_reference() -> None:
    rng = np.random.default_rng(4)
    docs = [
        [VOCAB[int(i)] for i in rng.integers(0, len(VOCAB), size=int(rng.integers(0, 30)))]
        for _ in range(LARGE_DOCS)
    ]
    query = ["cat", "fish", "bird"]
    want = _oracle(query, docs, 1.2, 0.75)
    start = time.perf_counter()
    got = _solve(query, docs)
    elapsed = time.perf_counter() - start
    assert elapsed < LARGE_TIME_LIMIT_S, (
        "a large collection is too slow: count each document once, not once per query token "
        "per document"
    )
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12), SCORE_MISTAKE


def test_query_list_keeps_its_original_order() -> None:
    query = ["sat", "cat", "dog"]
    docs = [["cat", "sat"], ["dog"], ["cat", "cat", "dog"]]
    before = list(query)
    bm25_scores(query, docs)
    assert query == before, (
        "the caller's query list was reordered: read the query tokens without sorting them in place"
    )
