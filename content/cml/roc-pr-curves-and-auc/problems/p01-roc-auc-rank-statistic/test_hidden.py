"""Hidden tests: scikit-learn and pair-count oracles, invariances, ties and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import math
import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score
from submission import roc_auc

from dogma_data import load, rng, sample_rows, synth

VALUE_MISTAKE = (
    "AUC differs from the reference: it is the share of positive-negative pairs ordered "
    "correctly, with a tie counted as one half"
)


WEAK_FEATURES = (9, 11, 14, 18)  # fractal dimension; texture, smoothness and symmetry errors


def _real_normal(gen, size=None, loc=0.0, scale=1.0):
    """Real numbers where this test used to draw from a normal distribution.

    Standardised breast_cancer measurements (mean 0 and standard deviation 1 over the dataset,
    skewed, with a few values far out), drawn by ``gen``. A vector is one real measurement
    over that many tumours, a matrix is real tumours by real measurements, and any other
    shape is filled from whole rows. ``loc`` and ``scale`` shift and stretch them as before.
    """
    shape = () if size is None else tuple(int(side) for side in np.atleast_1d(size))
    total = int(np.prod(shape)) if shape else 1
    if len(shape) == 1 and 1 <= shape[0] <= 569:
        values = sample_rows("breast_cancer", gen, shape[0], 1, standardize=True)[0][:, 0]
    elif len(shape) == 2 and 1 <= shape[0] <= 569 and 1 <= shape[1] <= 30:
        values = sample_rows("breast_cancer", gen, shape[0], shape[1], standardize=True)[0]
    else:
        rows = max(1, min(569, -(-total // 30)))
        drawn = sample_rows("breast_cancer", gen, rows, standardize=True)[0]
        values = np.resize(drawn.ravel(), total).reshape(shape)
    values = values * scale + loc
    return values if shape else float(values)


def _scored_tumours(gen, n, tied):
    """Real diagnoses of n breast_cancer tumours and one real feature as the score.

    With ``tied`` the score is a weak predictor rounded to whole standardised units, so many
    tumours share a score and tied tumours differ in diagnosis. Otherwise it is a randomly
    chosen feature as measured. A draw with a single class is skipped.
    """
    while True:
        features, target = sample_rows("breast_cancer", gen, n, standardize=True)
        if 0.0 < target.mean() < 1.0:
            break
    if tied:
        return target.astype(np.int64), np.round(features[:, int(gen.choice(WEAK_FEATURES))])
    return target.astype(np.int64), features[:, int(gen.integers(features.shape[1]))]


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _solve(y_true, scores):
    """Call the function on input that must be accepted. A crash is a contract mistake."""
    try:
        value = roc_auc(y_true, scores)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the rank sum and the pair formula"
        )
    assert isinstance(value, float), "return ROC AUC as a Python float"
    return value


def _pair_count(y_true, scores):
    """Independent route: count every positive-negative pair, a tie counting as one half."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    total = 0.0
    for p in pos:
        for q in neg:
            total += 1.0 if p > q else (0.5 if p == q else 0.0)
    return total / (len(pos) * len(neg))


@pytest.mark.parametrize(
    "seed", range(6), ids=[f"tied-{s}" for s in range(3)] + [f"continuous-{s}" for s in range(3)]
)
def test_matches_scikit_learn_on_tied_and_continuous_scores(seed):
    gen = rng(f"roc-auc-p01:scikit-learn:{seed}")
    n = int(gen.integers(20, 300))
    y, scores = _scored_tumours(gen, n, tied=seed < 3)
    expected = roc_auc_score(y, scores)
    assert math.isclose(_solve(y, scores), expected, rel_tol=1e-9, abs_tol=1e-12), VALUE_MISTAKE


@pytest.mark.parametrize("seed", range(4), ids=[f"pairs-{s}" for s in range(4)])
def test_matches_direct_pair_count(seed):
    gen = rng(f"roc-auc-p01:pair-count:{seed}")
    n = int(gen.integers(4, 40))
    y, scores = _scored_tumours(gen, n, tied=True)
    expected = _pair_count(y, scores)
    assert math.isclose(_solve(y, scores), expected, rel_tol=1e-12, abs_tol=1e-12), VALUE_MISTAKE


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 1), min_size=2, max_size=30),
    st.lists(st.integers(-3, 3), min_size=2, max_size=30),
)
def test_reversing_the_scores_gives_one_minus_auc(labels, grid):
    n = min(len(labels), len(grid))
    y = labels[:n]
    if sum(y) in (0, n):
        return
    s = [float(v) for v in grid[:n]]
    forward = _solve(y, s)
    reversed_ = _solve(y, [-v for v in s])
    assert math.isclose(forward + reversed_, 1.0, rel_tol=1e-12, abs_tol=1e-12), (
        "negating the scores must map AUC to 1 - AUC: check the pair direction and the tie rule"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 1), min_size=2, max_size=30),
    st.lists(st.integers(-50, 50), min_size=2, max_size=30),
)
def test_strictly_increasing_transform_keeps_the_auc(labels, grid):
    n = min(len(labels), len(grid))
    y = labels[:n]
    if sum(y) in (0, n):
        return
    s = [v / 8 for v in grid[:n]]  # exact binary fractions: exp keeps distinct values distinct
    assert math.isclose(
        _solve(y, s), _solve(y, [math.exp(v) for v in s]), rel_tol=1e-12, abs_tol=1e-12
    ), "AUC depends only on the ranking, so a strictly increasing map must not change it"


def test_tied_scores_count_as_half_for_all_tied_data_with_positions_mixed():
    got = _solve([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert math.isclose(got, 0.5, rel_tol=1e-12), (
        "tied scores must share their average rank: a position-based ranking gives a wrong AUC here"
    )


def test_a_tie_between_a_positive_and_a_negative_is_half_a_pair():
    got = _solve([0, 1], [0.3, 0.3])
    assert math.isclose(got, 0.5, rel_tol=1e-12), "a single tied pair must give AUC 0.5"


def test_perfect_and_inverted_rankings_give_one_and_zero():
    assert _solve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0, "a perfect ranking must give AUC 1.0"
    assert _solve([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0, (
        "an inverted ranking must give AUC 0.0"
    )


@pytest.mark.parametrize(
    ("y_true", "scores", "mistake"),
    [
        (
            [0, 0, 0],
            [0.1, 0.5, 0.9],
            "all-negative input was accepted: raise ValueError with no positives",
        ),
        ([1, 1], [0.1, 0.5], "all-positive input was accepted: raise ValueError with no negatives"),
        ([], [], "empty input was accepted: raise ValueError when there are no samples"),
        (
            [0, 2, 1, 0, 1],
            [0.1, 0.5, 0.9, 0.2, 0.4],
            "labels other than 0 and 1 were accepted: raise ValueError for them",
        ),
        ([0, 1, 1], [0.1, float("nan"), 0.9], "NaN scores were accepted: raise ValueError for NaN"),
        ([0, 1, 1], [0.1, 0.9], "unequal lengths were accepted: raise ValueError for them"),
        ([0, 1], [0.1, float("inf")], "infinite scores were accepted: raise ValueError"),
        ([[0, 1]], [[0.1, 0.9]], "matrix input was accepted: both inputs must be one-dimensional"),
    ],
    ids=[
        "no-positives",
        "no-negatives",
        "empty",
        "bad-label",
        "nan-score",
        "length-mismatch",
        "infinite-score",
        "matrix",
    ],
)
def test_invalid_input_raises_value_error(y_true, scores, mistake):
    _expect_value_error(lambda: roc_auc(y_true, scores), mistake)


def test_large_input_is_ranked_quickly_and_matches_scikit_learn():
    rng = np.random.default_rng(9)
    n = 300_000
    # Real diagnoses and real measurements of the 569 tumours, repeated to the length asked
    # for: the values tie in thousands, which is what a large real batch looks like.
    y = np.resize(load("breast_cancer").y.astype(np.int64), n)
    scores = _real_normal(rng, n)
    expected = roc_auc_score(y, scores)
    start = time.perf_counter()
    got = _solve(y, scores)
    elapsed = time.perf_counter() - start
    assert elapsed < 10, (
        "ranking 300,000 scores is too slow: use one sort, not pairwise comparisons"
    )
    assert math.isclose(got, expected, rel_tol=1e-9), VALUE_MISTAKE


def test_inputs_are_not_mutated():
    y = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.4, 0.9, 0.2, 0.7, 0.5])
    before_y, before_scores = y.copy(), scores.copy()
    _solve(y, scores)
    assert np.array_equal(y, before_y), "y_true was reordered or changed: work on sorted copies"
    assert np.array_equal(scores, before_scores), (
        "scores were reordered or changed: sort a copy, never the caller's array"
    )


def test_two_dimensional_labels_with_one_dimensional_scores_raise_value_error():
    labels = np.array([[0, 0], [1, 0], [0, 1]])
    scores = np.array([0.1, 0.5, 0.9])
    _expect_value_error(
        lambda: roc_auc(labels, scores),
        "a two-dimensional label array was accepted or crashed: y_true must be one-dimensional, "
        "so raise ValueError",
    )


PROBLEM = "cml/roc-pr-curves-and-auc/p01-roc-auc-rank-statistic"


@pytest.fixture(scope="module")
def breast_cancer():
    return load("breast_cancer")


def test_matches_scikit_learn_on_raw_breast_cancer_features(breast_cancer):
    names = list(breast_cancer.feature_names)
    malignant = 1 - np.asarray(breast_cancer.y).astype(int)  # 1 = malignant, the positive class
    gen = rng(f"{PROBLEM}:breast-cancer-full:0")
    for name in ("worst concave points", "area error", "mean texture"):
        column = breast_cancer.X[:, names.index(name)]
        expected = roc_auc_score(malignant, column)
        assert math.isclose(_solve(malignant, column), expected, rel_tol=1e-9, abs_tol=1e-12), (
            VALUE_MISTAKE
        )
    # Rare positives: keep 18 malignant rows and every benign row, a ratio near 1:20.
    positives = gen.choice(np.flatnonzero(malignant == 1), size=18, replace=False)
    rows = np.concatenate([positives, np.flatnonzero(malignant == 0)])
    for name in ("worst concave points", "area error"):
        column = breast_cancer.X[rows, names.index(name)]
        expected = roc_auc_score(malignant[rows], column)
        assert math.isclose(
            _solve(malignant[rows], column), expected, rel_tol=1e-9, abs_tol=1e-12
        ), VALUE_MISTAKE


def test_only_the_order_and_equality_of_scores_matter_at_tiny_magnitudes(breast_cancer):
    names = list(breast_cancer.feature_names)
    malignant = 1 - np.asarray(breast_cancer.y).astype(int)
    column = breast_cancer.X[:, names.index("worst concave points")]
    base = _solve(malignant, column)
    tiny = _solve(malignant, column * 1e-9)
    assert math.isclose(base, tiny, rel_tol=1e-12, abs_tol=1e-12), (
        "AUC changed when the scores were scaled by a tiny constant: ties are exact equality, "
        "so no absolute tolerance may merge distinct scores"
    )
    gen = rng(f"{PROBLEM}:tiny-scale:0")
    y = synth.imbalanced_labels(gen, 4000, positive_rate=0.02)
    scores = synth.informative_scores(gen, y, auc=0.9, decimals=4)
    assert math.isclose(
        _solve(y, scores * 1e-9), roc_auc_score(y, scores), rel_tol=1e-9, abs_tol=1e-12
    ), "AUC changed when the scores were scaled by a tiny constant: ties are exact equality"
