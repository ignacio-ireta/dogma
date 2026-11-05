"""Public tests: the examples from problem.md, the tie rule, k, and one degenerate case.

The learner's code is imported from submission.py. Tests never import solution.py.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from submission import ndcg_at_k

LOG2_3 = math.log2(3)


def _solve(relevance: Any, scores: Any, k: int | None = None) -> float:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        return ndcg_at_k(relevance, scores, k)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the ranking, the tie groups and the "
            "all-zero case"
        )


def _expect_value_error(call: Any, mistake: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_ideal_order_scores_one() -> None:
    got = _solve([0, 1], [0.2, 0.9])
    assert math.isclose(got, 1.0, rel_tol=1e-12), (
        "an ideal ranking must score 1.0: the ideal DCG must sort the relevance itself, descending"
    )


def test_tied_scores_share_their_mean_gain_over_their_positions() -> None:
    # Items 1 and 2 tie at positions 1 and 2: each takes the mean gain 1.5 there.
    expected = (1.5 * (1 + 1 / LOG2_3) + 1 * 0.5) / (3 + 1 / LOG2_3)
    got = _solve([0, 3, 1], [0.7, 0.7, 0.2])
    assert math.isclose(got, expected, rel_tol=1e-9), (
        "tied items must share the average gain over the positions they occupy; "
        "breaking the tie by input position gives a different value"
    )


def test_k_keeps_only_the_top_positions_in_dcg_and_ideal() -> None:
    expected = (1 + 3 / LOG2_3) / (3 + 2 / LOG2_3)
    got = _solve([1, 3, 2], [0.9, 0.8, 0.1], k=2)
    assert math.isclose(got, expected, rel_tol=1e-9), (
        "with k=2 only the first two positions count, in the DCG and in the ideal DCG"
    )


def test_all_zero_relevance_gives_zero() -> None:
    assert _solve([0, 0], [0.3, 0.1]) == 0.0, (
        "when every relevance is 0 the ideal DCG is 0: return 0.0, not NaN or an error"
    )


def test_mismatched_lengths_raise_value_error() -> None:
    _expect_value_error(
        lambda: ndcg_at_k([1, 0], [0.5]),
        "unequal lengths were accepted: raise ValueError when the two inputs differ in length",
    )
