"""Public tests: the examples from problem.md, the empty-query case and one argument error.

The learner's code is imported from submission.py. Tests never import solution.py.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from submission import bm25_scores

DOCS = [["cat", "sat"], ["dog"], ["cat", "cat", "dog"]]
IDF_CAT = math.log(1.6)  # N = 3, n(cat) = 2: ln(1 + 1.5 / 2.5)


def _solve(query: Any, docs: Any, **kwargs: Any) -> list[float]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        got = bm25_scores(query, docs, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the idf, the saturation and the "
            "average length"
        )
    assert isinstance(got, list) and len(got) == len(docs), (
        "return one score per document, as a list, in input order"
    )
    return got


def test_single_term_scores_follow_idf_saturation_and_length() -> None:
    got = _solve(["cat"], DOCS)
    assert math.isclose(got[0], IDF_CAT, rel_tol=1e-9), (
        "the first document has |D| = avgdl and one occurrence, so its score is exactly the idf"
    )
    expected_third = IDF_CAT * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert math.isclose(got[2], expected_third, rel_tol=1e-9), (
        "two occurrences in a longer than average document: saturation and length both apply"
    )
    assert got[1] == 0.0, "a document without the query token must score 0.0"


def test_unseen_query_token_adds_nothing() -> None:
    assert _solve(["cat", "zebra"], DOCS) == pytest.approx(_solve(["cat"], DOCS)), (
        "a query token that is in no document contributes 0 to every score"
    )


def test_k1_zero_gives_each_matching_document_its_idf() -> None:
    got = _solve(["cat"], DOCS, k1=0.0)
    assert math.isclose(got[0], IDF_CAT, rel_tol=1e-9), (
        "with k1 = 0 the term frequency is ignored, so a match scores exactly the idf"
    )
    assert math.isclose(got[2], IDF_CAT, rel_tol=1e-9), (
        "with k1 = 0 two occurrences score the same as one"
    )


def test_empty_query_scores_zero_for_every_document() -> None:
    assert _solve([], DOCS) == [0.0, 0.0, 0.0], "an empty query matches nothing: all scores are 0.0"


def test_negative_k1_raises_value_error() -> None:
    try:
        bm25_scores(["cat"], DOCS, k1=-1.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative k1 must raise ValueError (raised {type(exc).__name__})")
    pytest.fail("a negative k1 was accepted: raise ValueError, k1 must not be negative")
