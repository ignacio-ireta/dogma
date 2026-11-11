"""Hidden tests: scikit-learn and brute-force oracles, invariances, ties, edge cases and speed.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

from __future__ import annotations

import itertools
import math
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import ArrayLike
from sklearn.metrics import ndcg_score
from submission import ndcg_at_k

from dogma_data import load, rng

LARGE_N = 200_000
LARGE_TIME_LIMIT_S = 10.0
SMALL_TIME_LIMIT_S = 5.0

VALUE_MISTAKE = (
    "NDCG differs from the reference: sort by score descending, gain = relevance, discount "
    "1 / log2(rank + 1), and average the gain over each group of tied scores"
)


class _TooSlow(BaseException):
    """Raised inside the learner's call when it runs past its time box."""


@contextmanager
def _time_box(seconds: float) -> Iterator[None]:
    """Interrupt the learner's code after ``seconds``. Only the main thread can use signals."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    def _expired(signum: int, frame: Any) -> None:
        raise _TooSlow

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


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


def _solve(relevance: Any, scores: Any, k: int | None = None, *, limit: float = 5.0) -> float:
    """Call the learner on input that must be accepted. A crash is a contract mistake."""
    try:
        with _time_box(limit):
            value = ndcg_at_k(relevance, scores, k)
    except _TooSlow:
        _fail("too slow on a large list: rank once with a sort, do not compare every pair")
    except Exception as exc:
        _fail(
            f"valid input raised {type(exc).__name__}: check the ranking, the tie groups and the "
            "all-zero case"
        )
    if not isinstance(value, float):
        _fail("return NDCG as a Python float")
    return value


def _sklearn(relevance: ArrayLike, scores: ArrayLike, k: int | None = None) -> float:
    """Oracle: scikit-learn's default, which averages the gains over tied scores."""
    return float(
        ndcg_score([np.asarray(relevance, dtype=float)], [np.asarray(scores, dtype=float)], k=k)
    )


def _expected_by_enumeration(
    relevance: Sequence[float], scores: Sequence[float], k: int | None
) -> float:
    """The definition itself: DCG averaged over every order of the tied items, over IDCG."""
    n = len(relevance)
    cutoff = n if k is None else min(k, n)
    discount = [1.0 / math.log2(r + 1) for r in range(1, n + 1)]
    groups: dict[float, list[int]] = {}
    for index, score in enumerate(scores):
        groups.setdefault(score, []).append(index)
    per_group = [list(itertools.permutations(groups[s])) for s in sorted(groups, reverse=True)]
    total = 0.0
    count = 0
    for choice in itertools.product(*per_group):
        order = [index for block in choice for index in block]
        total += sum(relevance[order[r]] * discount[r] for r in range(cutoff))
        count += 1
    dcg = total / count
    ideal = sorted(relevance, reverse=True)
    idcg = sum(ideal[r] * discount[r] for r in range(cutoff))
    return 0.0 if idcg == 0 else dcg / idcg


@pytest.mark.parametrize(
    "seed", range(6), ids=[f"tied-{s}" for s in range(3)] + [f"continuous-{s}" for s in range(3)]
)
def test_matches_scikit_learn_on_tied_and_continuous_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 120))
    relevance = rng.integers(0, 5, size=n).astype(float)
    scores = rng.integers(0, 6, size=n) / 5.0 if seed < 3 else rng.normal(size=n)
    k = None if seed % 2 == 0 else int(rng.integers(1, n + 1))
    expected = _sklearn(relevance, scores, k)
    assert math.isclose(_solve(relevance, scores, k), expected, rel_tol=1e-9, abs_tol=1e-12), (
        VALUE_MISTAKE
    )


@pytest.mark.parametrize("seed", range(5), ids=[f"enumerated-{s}" for s in range(5)])
def test_matches_the_enumerated_tie_expectation(seed: int) -> None:
    rng = np.random.default_rng(40 + seed)
    for _ in range(12):
        n = int(rng.integers(2, 7))
        relevance = [float(v) for v in rng.integers(0, 4, size=n)]
        scores = [float(v) for v in rng.integers(0, 3, size=n)]  # heavy ties
        k = None if rng.random() < 0.5 else int(rng.integers(1, n + 1))
        expected = _expected_by_enumeration(relevance, scores, k)
        assert math.isclose(_solve(relevance, scores, k), expected, rel_tol=1e-9, abs_tol=1e-12), (
            VALUE_MISTAKE
        )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 4), min_size=2, max_size=12),
    st.lists(st.integers(-3, 3), min_size=2, max_size=12),
)
def test_reordering_the_items_changes_nothing(relevance: list[int], grid: list[int]) -> None:
    n = min(len(relevance), len(grid))
    rel = [float(v) for v in relevance[:n]]
    scores = [float(v) for v in grid[:n]]
    perm = list(range(n))
    perm.reverse()
    forward = _solve(rel, scores)
    shuffled = _solve([rel[i] for i in perm], [scores[i] for i in perm])
    assert math.isclose(forward, shuffled, rel_tol=1e-12, abs_tol=1e-12), (
        "the value depends on the order of the input items: ties must not be broken by position"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 4), min_size=2, max_size=12),
    st.lists(st.integers(-40, 40), min_size=2, max_size=12),
)
def test_strictly_increasing_transform_of_scores_changes_nothing(
    relevance: list[int], grid: list[int]
) -> None:
    n = min(len(relevance), len(grid))
    rel = [float(v) for v in relevance[:n]]
    scores = [v / 8 for v in grid[:n]]  # exact binary fractions: exp keeps distinct values distinct
    assert math.isclose(
        _solve(rel, scores), _solve(rel, [math.exp(s) for s in scores]), rel_tol=1e-12
    ), "NDCG depends only on the ranking, so a strictly increasing map must not change it"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 4), min_size=2, max_size=10),
    st.lists(st.integers(-3, 3), min_size=2, max_size=10),
    st.sampled_from([0.5, 2.0, 7.0]),
)
def test_scaling_the_relevance_changes_nothing(
    relevance: list[int], grid: list[int], factor: float
) -> None:
    n = min(len(relevance), len(grid))
    rel = [float(v) for v in relevance[:n]]
    scores = [float(v) for v in grid[:n]]
    scaled = [factor * v for v in rel]
    assert math.isclose(_solve(rel, scores), _solve(scaled, scores), rel_tol=1e-12), (
        "NDCG is a ratio of two DCGs with the same gains, so a common scale must cancel"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 4), min_size=1, max_size=15),
    st.lists(st.integers(-3, 3), min_size=1, max_size=15),
)
def test_value_stays_in_the_unit_interval(relevance: list[int], grid: list[int]) -> None:
    n = min(len(relevance), len(grid))
    value = _solve([float(v) for v in relevance[:n]], [float(v) for v in grid[:n]])
    assert 0.0 <= value <= 1.0 + 1e-12, "NDCG must lie in [0, 1]: check the ideal normalisation"


def test_all_scores_tied_gives_the_mean_gain_over_all_positions() -> None:
    relevance = [3.0, 0.0, 1.0, 2.0]
    discount = [1 / math.log2(r + 1) for r in range(1, 5)]
    dcg = (sum(relevance) / 4) * sum(discount)
    idcg = sum(g * d for g, d in zip(sorted(relevance, reverse=True), discount, strict=True))
    assert math.isclose(_solve(relevance, [1.0] * 4), dcg / idcg, rel_tol=1e-12), (
        "when every score ties, each position gets the mean gain: average over the whole list"
    )


def test_all_zero_relevance_gives_zero_for_any_k() -> None:
    for k in (None, 1, 3, 50):
        got = _solve([0.0, 0.0, 0.0], [0.3, 0.1, 0.2], k)
        assert got == 0.0, "all-zero relevance must give 0.0, not NaN or an error, whatever k is"


def test_k_larger_than_n_is_the_whole_list() -> None:
    relevance, scores = [3.0, 1.0, 2.0, 0.0], [0.1, 0.9, 0.5, 0.3]
    assert math.isclose(_solve(relevance, scores, 50), _solve(relevance, scores), rel_tol=1e-12), (
        "k larger than the list must mean the whole list, not an index error"
    )


def test_k_equal_to_n_and_none_agree() -> None:
    relevance, scores = [2.0, 0.0, 1.0], [0.4, 0.8, 0.6]
    assert math.isclose(_solve(relevance, scores, 3), _solve(relevance, scores), rel_tol=1e-12), (
        "k = n and k = None must give the same value"
    )


def test_single_item_is_a_valid_query() -> None:
    assert _solve([2.0], [5.0]) == 1.0, "one relevant item is already ideal: NDCG must be 1.0"
    assert _solve([0.0], [5.0]) == 0.0, "one irrelevant item has no ideal gain: NDCG must be 0.0"


def test_exact_rational_example_with_ties_matches_the_enumeration() -> None:
    relevance, scores = [0.0, 3.0, 1.0, 2.0, 2.0], [0.5, 0.5, 0.5, 0.1, 0.1]
    expected = _expected_by_enumeration(relevance, scores, 4)
    assert math.isclose(_solve(relevance, scores, 4), expected, rel_tol=1e-12), (
        "a tie group that straddles the cutoff must count only the positions within k"
    )


def test_perfect_order_with_ties_scores_one() -> None:
    assert math.isclose(_solve([3.0, 3.0, 1.0], [0.9, 0.9, 0.2]), 1.0, rel_tol=1e-12), (
        "tied items with the same relevance, ranked above the rest, give NDCG 1.0"
    )


@pytest.mark.parametrize(
    ("relevance", "scores", "k", "mistake"),
    [
        (
            [1.0, -1.0, 0.0],
            [1.0, 2.0, 3.0],
            None,
            "negative relevance was accepted: raise ValueError",
        ),
        ([1.0, 0.0], [1.0, float("nan")], None, "NaN scores were accepted: raise ValueError"),
        ([1.0, 0.0], [1.0, float("inf")], None, "infinite scores were accepted: raise ValueError"),
        ([1.0, 0.0], [1.0, 2.0], 0, "k = 0 was accepted: k must be at least 1"),
        ([], [], None, "empty input was accepted: raise ValueError when there are no items"),
        ([1.0, 0.0], [1.0], None, "unequal lengths were accepted: raise ValueError"),
        ([[1.0, 0.0]], [[1.0, 2.0]], None, "matrix input was accepted: both inputs must be 1-D"),
        ([1.0, float("nan")], [1.0, 2.0], None, "NaN relevance was accepted: raise ValueError"),
    ],
    ids=[
        "negative-relevance",
        "nan-score",
        "infinite-score",
        "k-zero",
        "empty",
        "length-mismatch",
        "matrix",
        "nan-relevance",
    ],
)
def test_invalid_input_raises_value_error(
    relevance: Any, scores: Any, k: int | None, mistake: str
) -> None:
    _expect_value_error(lambda: ndcg_at_k(relevance, scores, k), mistake)


def test_large_list_finishes_quickly_and_matches_scikit_learn() -> None:
    rng = np.random.default_rng(9)
    relevance = rng.integers(0, 4, size=LARGE_N).astype(float)
    scores = rng.normal(size=LARGE_N)
    expected = _sklearn(relevance, scores, 1000)
    start = time.perf_counter()
    got = _solve(relevance, scores, 1000, limit=LARGE_TIME_LIMIT_S)
    elapsed = time.perf_counter() - start
    assert elapsed < LARGE_TIME_LIMIT_S, (
        "a large list is too slow: rank with one sort and use prefix sums of the discounts"
    )
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12), VALUE_MISTAKE


def test_inputs_are_not_mutated() -> None:
    relevance = np.array([0.0, 3.0, 1.0, 2.0])
    scores = np.array([0.2, 0.9, 0.9, 0.4])
    before_relevance, before_scores = relevance.copy(), scores.copy()
    _solve(relevance, scores)
    assert np.array_equal(relevance, before_relevance), (
        "true_relevance was reordered or changed: work on sorted copies"
    )
    assert np.array_equal(scores, before_scores), (
        "scores were reordered or changed: sort a copy, never the caller's array"
    )


PROBLEM = "cml/ranking-metrics/p01-ndcg-at-k"


@pytest.fixture(scope="module")
def breast_cancer() -> Any:
    return load("breast_cancer")


def _graded_quartiles(column: np.ndarray) -> np.ndarray:
    """Grades 0 to 3 from the quartiles of a real column: the lowest quarter has grade 0."""
    cuts = np.quantile(column, [0.25, 0.5, 0.75])
    return np.searchsorted(cuts, column, side="right").astype(float)


@pytest.mark.parametrize("k", [5, 25, None], ids=["k-5", "k-25", "whole-list"])
def test_matches_scikit_learn_on_raw_breast_cancer_with_graded_relevance(
    breast_cancer: Any, k: int | None
) -> None:
    names = list(breast_cancer.feature_names)
    relevance = _graded_quartiles(breast_cancer.X[:, names.index("area error")])
    scores = breast_cancer.X[:, names.index("worst concave points")]  # natural ties
    expected = _sklearn(relevance, scores, k)
    assert math.isclose(_solve(relevance, scores, k), expected, rel_tol=1e-9, abs_tol=1e-12), (
        VALUE_MISTAKE
    )


def test_only_the_order_and_equality_of_scores_matter_at_tiny_magnitudes(
    breast_cancer: Any,
) -> None:
    names = list(breast_cancer.feature_names)
    data = breast_cancer
    relevance = _graded_quartiles(data.X[:, names.index("area error")])
    scores = data.X[:, names.index("worst concave points")]
    base = _solve(relevance, scores, 25)
    tiny = _solve(relevance, scores * 1e-9, 25)
    assert math.isclose(base, tiny, rel_tol=1e-12, abs_tol=1e-12), (
        "NDCG changed when the scores were scaled by a tiny constant: tied scores are exact "
        "equality, so no absolute tolerance may merge distinct score groups"
    )
    gen = rng(f"{PROBLEM}:tiny-scale:0")
    noisy = np.round(gen.normal(size=300), 3)  # rounded scores: real ties, small magnitudes
    grades = gen.integers(0, 4, size=300).astype(float)
    assert math.isclose(
        _solve(grades, noisy * 1e-9), _sklearn(grades, noisy), rel_tol=1e-9, abs_tol=1e-12
    ), "NDCG with tiny-magnitude scores must still average over exact tie groups only"


def test_a_matrix_of_scores_alone_raises_value_error() -> None:
    _expect_value_error(
        lambda: ndcg_at_k([1.0, 0.0], [[1.0, 2.0, 3.0], [0.5, 0.2, 0.1]], None),
        "a matrix of scores was accepted or failed with the wrong error: both inputs must be "
        "one-dimensional, so raise ValueError when only one of them is a matrix",
    )
