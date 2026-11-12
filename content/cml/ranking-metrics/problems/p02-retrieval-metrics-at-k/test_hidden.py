"""Hidden tests: a NumPy oracle, denominators, reciprocal rank beyond k, invariances and errors.

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
from submission import retrieval_metrics

from dogma_data import rng as data_rng
from dogma_data import sample_rows

KEYS = ("precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision_at_k")
LARGE_N = 200_000
LARGE_TIME_LIMIT_S = 5.0
POOL = [3, 17, 42, 1_204, 17_204, 905_113_208, 6_000_000_001, "doc-x", "doc-y", "doc-z"]


def _fail(why: str) -> None:
    raise AssertionError(why)


def _solve(ranked: Any, relevant: Any, k: Any) -> dict[str, float]:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        got = retrieval_metrics(ranked, relevant, k)
    except Exception as exc:
        _fail(
            f"valid input raised {type(exc).__name__}: check the hit counts, the denominators and "
            "the empty ranking"
        )
    if not isinstance(got, dict) or set(got) != set(KEYS):
        _fail("return a dict with exactly the four keys from problem.md")
    return got


def _expect_value_error(call: Callable[[], object], mistake: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        _fail(f"{mistake} (raised {type(exc).__name__} instead)")
    _fail(mistake)


def _oracle(ranked: list[Any], relevant: set[Any], k: int) -> dict[str, float]:
    """Independent computation: a hit vector and cumulative sums, with explicit denominators."""
    hits = np.array([item in relevant for item in ranked], dtype=float)
    n = len(hits)
    within = hits[:k]
    found_in_top = float(within.sum())
    precision = found_in_top / k
    recall = found_in_top / len(relevant)
    first = int(np.argmax(hits)) + 1 if hits.any() else None
    reciprocal = 0.0 if first is None else 1.0 / first
    running_precision = np.cumsum(hits) / np.arange(1, n + 1) if n else np.zeros(0)
    top_len = min(k, n)
    ap_sum = float(np.sum(running_precision[:top_len] * hits[:top_len]))
    average = ap_sum / min(k, len(relevant))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal,
        "average_precision_at_k": average,
    }


MISTAKE_BY_KEY = {
    "precision_at_k": (
        "precision_at_k differs from the reference: count the hits in the top k and divide by k, "
        "even when the list is short"
    ),
    "recall_at_k": (
        "recall_at_k differs from the reference: divide the hits in the top k by the number of "
        "distinct relevant ids"
    ),
    "reciprocal_rank": (
        "reciprocal_rank differs from the reference: use the first hit in the whole ranking, not "
        "only the top k, and count positions from 1"
    ),
    "average_precision_at_k": (
        "average_precision_at_k differs from the reference: add precision only at relevant "
        "positions within k, and divide by min(k, |relevant|)"
    ),
}


def _check_against(ranked: list[Any], relevant: set[Any], k: int) -> None:
    want = _oracle(ranked, relevant, k)  # computed before the learner runs
    got = _solve(list(ranked), list(relevant), k)
    for key in KEYS:
        if not math.isclose(got[key], want[key], rel_tol=1e-9, abs_tol=1e-12):
            _fail(MISTAKE_BY_KEY[key])


@pytest.mark.parametrize("seed", range(20), ids=[f"random-{s}" for s in range(20)])
def test_matches_the_numpy_reference_on_random_rankings(seed: int) -> None:
    # A real ranking: 22 california_housing block groups with the ids of the pool. The relevant
    # ones are those whose house value is above 2.6 (about the top quarter of the table), or
    # the most valuable one when none is. Even seeds rank by median income, a good predictor;
    # odd seeds rank by house age, a poor one, so the first relevant id can sit deep in the
    # list. The ranking is cut at a random length, so relevant ids can be missing from it.
    rng = data_rng(f"retrieval-metrics-p02:ranking:{seed}")
    pool = POOL + [f"doc-{i}" for i in range(12)]
    features, value = sample_rows("california_housing", rng, len(pool))
    size = int(rng.integers(0, 15))
    ranker = features[:, 1] if seed % 2 else features[:, 0]
    ranked = [pool[i] for i in np.argsort(-ranker, kind="stable")[:size]]
    relevant = {pool[i] for i in np.flatnonzero(value > 2.6)} or {pool[int(np.argmax(value))]}
    k = int(rng.integers(1, 18))  # sometimes larger than the list
    _check_against(ranked, relevant, k)


def test_average_precision_is_one_when_the_top_k_are_all_relevant() -> None:
    got = _solve(["a", "b", "c", "d"], ["a", "b", "c", "d", "e"], 2)
    assert math.isclose(got["average_precision_at_k"], 1.0, rel_tol=1e-12), (
        "AP@k is 1.0 when the top k are all relevant, even if |relevant| > k: the divisor is "
        "min(k, |relevant|), not |relevant|"
    )
    assert math.isclose(got["recall_at_k"], 0.4, rel_tol=1e-12), (
        "recall must divide retrieved relevant ids by the total number of relevant ids"
    )


def test_reciprocal_rank_is_not_cut_at_k() -> None:
    got = _solve(["x", "y", "z", "a"], ["a"], 2)
    assert math.isclose(got["reciprocal_rank"], 0.25, rel_tol=1e-12), (
        "reciprocal rank must look for the first relevant item in the whole ran"
        "king, beyond the cutoff too"
    )
    assert got["precision_at_k"] == 0.0 and got["average_precision_at_k"] == 0.0, (
        "precision and average precision must ignore relevant items past the cutoff"
    )


def test_average_precision_divisor_is_min_of_k_and_relevant_count() -> None:
    got = _solve(["a", "b"], ["a", "b", "c"], 10)
    assert math.isclose(got["average_precision_at_k"], 2 / 3, rel_tol=1e-12), (
        "average precision must include all relevant ids in its stated divisor,"
        " including ones never ranked"
    )
    assert math.isclose(got["precision_at_k"], 0.2, rel_tol=1e-12), (
        "precision must divide by k even when the retrieved list is shorter"
    )


def test_duplicate_relevant_ids_are_counted_once() -> None:
    got = _solve(["a", "b", "c"], ["a", "a", "b"], 1)
    assert math.isclose(got["recall_at_k"], 0.5, rel_tol=1e-12), (
        "relevant ids are a set: the duplicate 'a' must not count twice in |relevant|"
    )


def test_empty_ranking_gives_zeros_and_is_valid() -> None:
    got = _solve([], ["a"], 3)
    assert all(got[key] == 0.0 for key in KEYS), (
        "an empty ranking retrieves nothing: every metric is 0.0, and no error is raised"
    )


def test_perfect_ranking_gives_ones() -> None:
    got = _solve([905_113_208, 17, 3], [3, 17, 905_113_208], 3)
    assert all(math.isclose(got[key], 1.0, rel_tol=1e-12) for key in KEYS), (
        "a ranking with every relevant id first gives 1.0 for all four metrics at k = |relevant|"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 30), min_size=1, max_size=12, unique=True),
    st.lists(st.integers(0, 30), min_size=1, max_size=8, unique=True),
    st.integers(1, 12),
    st.integers(1, 6),
)
def test_appending_irrelevant_items_changes_no_metric(
    ranked: list[int], relevant: list[int], k: int, extra: int
) -> None:
    rel = set(relevant)
    filler = [1000 + i for i in range(extra)]  # never in the relevant set
    base = _solve(list(ranked), list(rel), k)
    longer = _solve(list(ranked) + filler, list(rel), k)
    for key in KEYS:
        assert math.isclose(base[key], longer[key], rel_tol=1e-12, abs_tol=1e-12), (
            f"{key} changed when only irrelevant ids were appended: a denominator uses the list "
            "length, not k or |relevant|"
        )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 30), min_size=1, max_size=12, unique=True),
    st.lists(st.integers(0, 30), min_size=1, max_size=8, unique=True),
)
def test_recall_never_decreases_as_k_grows(ranked: list[int], relevant: list[int]) -> None:
    values = [
        _solve(list(ranked), list(relevant), k)["recall_at_k"] for k in range(1, len(ranked) + 2)
    ]
    assert all(a <= b + 1e-12 for a, b in pairwise(values)), (
        "recall at k cannot fall as k grows: count the hits in the top k, do not reset per k"
    )
    assert all(0.0 <= v <= 1.0 for v in values), "recall must stay in [0, 1]"


def test_invalid_input_raises_value_error() -> None:
    _expect_value_error(
        lambda: retrieval_metrics(["a", "b", "a"], ["a"], 2),
        "a repeated ranked id was accepted: raise ValueError, each document is ranked once",
    )
    _expect_value_error(
        lambda: retrieval_metrics(["a", "b"], [], 2),
        "an empty relevant set was accepted: raise ValueError, because recall is undefined",
    )
    _expect_value_error(
        lambda: retrieval_metrics(["a", "b"], ["a"], 0),
        "k = 0 was accepted: k must be an integer of at least 1",
    )


def test_inputs_are_not_mutated() -> None:
    ranked = [905_113_208, 17, 3, 42]
    relevant = {17, 42}
    before_ranked = list(ranked)
    before_relevant = set(relevant)
    _solve(ranked, relevant, 3)
    assert ranked == before_ranked, (
        "ranked_ids was reordered or changed: copy the list before using it"
    )
    assert relevant == before_relevant, "relevant_ids was changed: do not edit the caller's set"


def test_long_ranking_finishes_quickly_and_matches_the_reference() -> None:
    rng = np.random.default_rng(3)
    ranked = [int(v) for v in rng.permutation(LARGE_N)]
    relevant = {int(v) for v in rng.choice(LARGE_N, size=500, replace=False)}
    want = _oracle(ranked, relevant, 100)
    start = time.perf_counter()
    got = _solve(ranked, relevant, 100)
    elapsed = time.perf_counter() - start
    assert elapsed < LARGE_TIME_LIMIT_S, (
        "a long ranking is too slow: make one pass over the ranking, not one pass per position"
    )
    for key in KEYS:
        assert math.isclose(got[key], want[key], rel_tol=1e-9, abs_tol=1e-12), (
            f"{key} differs from the reference on a long ranking: check the hit counts"
        )
