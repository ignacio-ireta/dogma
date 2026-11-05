"""Public tests: the examples from problem.md, a short list, and one degenerate case.

The learner's code is imported from submission.py. Tests never import solution.py.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from submission import retrieval_metrics

KEYS = {"precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision_at_k"}


def _solve(ranked: Any, relevant: Any, k: Any) -> dict[str, float]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        got = retrieval_metrics(ranked, relevant, k)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the hit counts, the denominators and "
            "the empty ranking"
        )
    assert isinstance(got, dict) and set(got) == KEYS, (
        "return a dict with exactly the four keys: precision_at_k, recall_at_k, "
        "reciprocal_rank, average_precision_at_k"
    )
    return got


def test_first_example_returns_all_four_values() -> None:
    got = _solve(["d3", "d1", "d7", "d2"], ["d1", "d2", "d9"], 3)
    assert math.isclose(got["precision_at_k"], 1 / 3, rel_tol=1e-9), (
        "precision at 3 is the hits in the top 3 divided by 3"
    )
    assert math.isclose(got["recall_at_k"], 1 / 3, rel_tol=1e-9), (
        "recall at 3 is the hits in the top 3 divided by the number of relevant ids"
    )
    assert math.isclose(got["reciprocal_rank"], 0.5, rel_tol=1e-9), (
        "reciprocal rank is 1 over the position of the first relevant id, here position 2"
    )
    assert math.isclose(got["average_precision_at_k"], 0.5 / 3, rel_tol=1e-9), (
        "average precision divides the precision sum by min(k, |relevant|), here min(3, 3)"
    )


def test_precision_divides_by_k_even_when_the_list_is_shorter() -> None:
    got = _solve(["a", "b", "c"], ["a", "c"], 5)
    assert math.isclose(got["precision_at_k"], 0.4, rel_tol=1e-9), (
        "precision divides by k, not by the list length: 2 hits over k = 5 gives 0.4"
    )
    assert math.isclose(got["average_precision_at_k"], (1 + 2 / 3) / 2, rel_tol=1e-9), (
        "average precision divides by min(k, |relevant|) = min(5, 2) = 2"
    )


def test_nothing_relevant_retrieved_gives_zeros() -> None:
    got = _solve(["x", "y"], ["z"], 2)
    assert all(value == 0.0 for value in got.values()), (
        "with no relevant id retrieved every metric must be 0.0, without an error"
    )


def test_repeated_ranked_id_raises_value_error() -> None:
    try:
        retrieval_metrics(["a", "b", "a"], ["a"], 2)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a repeated ranked id must raise ValueError (raised {type(exc).__name__})")
    pytest.fail("a repeated ranked id was accepted: raise ValueError, each document is ranked once")
