"""Hidden tests: scikit-learn oracle, an explicit threshold sum, tie rules and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score
from submission import average_precision

from dogma_data import load, rng, sample_rows

VALUE_MISTAKE = (
    "AP differs from the reference: a step sum over the distinct thresholds of (recall gained) "
    "times (precision at that threshold), highest threshold first"
)


WEAK_FEATURES = (9, 11, 14, 18)  # fractal dimension; texture, smoothness and symmetry errors


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
        value = average_precision(y_true, scores)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the threshold sweep and the weights"
        )
    assert isinstance(value, float), "return average precision as a Python float"
    return value


def _threshold_sum(y_true, scores):
    """Independent route: loop over the distinct thresholds, highest first, with explicit counts."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    total = 0.0
    previous_recall = 0.0
    for t in sorted(set(s.tolist()), reverse=True):
        flagged = s >= t
        tp = int(np.sum(flagged & (y == 1)))
        precision = tp / int(np.sum(flagged))
        recall = tp / n_pos
        total += (recall - previous_recall) * precision
        previous_recall = recall
    return total


@pytest.mark.parametrize("seed", range(6), ids=[f"case-{s}" for s in range(6)])
def test_matches_scikit_learn_on_tied_and_continuous_scores(seed):
    gen = rng(f"average-precision-p02:scikit-learn:{seed}")
    n = int(gen.integers(5, 80))
    y, scores = _scored_tumours(gen, n, tied=bool(seed % 2))
    expected = average_precision_score(y, scores)
    assert math.isclose(_solve(y, scores), expected, rel_tol=1e-9, abs_tol=1e-12), VALUE_MISTAKE


@pytest.mark.parametrize("seed", range(4), ids=[f"sum-{s}" for s in range(4)])
def test_matches_explicit_threshold_sum(seed):
    gen = rng(f"average-precision-p02:threshold-sum:{seed}")
    n = int(gen.integers(4, 40))
    y, scores = _scored_tumours(gen, n, tied=True)
    expected = _threshold_sum(y, scores)
    assert math.isclose(_solve(y, scores), expected, rel_tol=1e-12, abs_tol=1e-12), VALUE_MISTAKE


def test_the_explicit_example_is_five_sixths_not_the_trapezoidal_value():
    got = _solve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert math.isclose(got, 5 / 6, rel_tol=1e-12), (
        "AP is a step sum: (0.5)(1) + (0)(0.5) + (0.5)(2/3) + (0)(0.5) = 5/6, not a trapezoid"
    )


def test_tied_scores_are_one_threshold_not_split():
    got = _solve([0, 1, 0], [0.0, 0.0, 0.0])
    assert math.isclose(got, 1 / 3, rel_tol=1e-12), (
        "tied scores were split: one threshold covers the whole tie, so AP equals the positive rate"
    )


def test_a_tie_between_positive_and_negative_is_not_resolved_by_position():
    got_a = _solve([1, 0], [0.5, 0.5])
    got_b = _solve([0, 1], [0.5, 0.5])
    assert math.isclose(got_a, 0.5, rel_tol=1e-12) and math.isclose(got_b, 0.5, rel_tol=1e-12), (
        "a tie must not depend on the order of the samples: the threshold covers both"
    )


def test_perfect_ranking_gives_exactly_one():
    assert _solve([0, 1, 0, 1, 1], [0.1, 0.9, 0.2, 0.8, 0.7]) == 1.0, (
        "a perfect ranking must give AP 1.0: the first positive's recall step counts from 0"
    )


def test_all_positive_samples_have_average_precision_one():
    assert _solve([1, 1, 1], [0.2, 0.7, 0.7]) == 1.0, (
        "AP is defined with only positives and equals one; only missing positives are rejected"
    )


def test_ascending_order_is_not_used():
    got = _solve([1, 0, 0], [0.9, 0.5, 0.1])
    assert math.isclose(got, 1.0, rel_tol=1e-12), (
        "AP must sweep from the highest threshold down; this ranking puts the positive first"
    )


def test_weights_are_the_recall_gained_not_a_plain_mean_of_precisions():
    got = _solve([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5])
    assert math.isclose(got, 5 / 6, rel_tol=1e-12), (
        "AP weights each precision by the recall gained at its threshold, not by one per threshold"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 1), min_size=1, max_size=25),
    st.lists(st.integers(0, 6), min_size=1, max_size=25),
)
def test_constant_scores_give_the_positive_rate(labels, grid):
    n = min(len(labels), len(grid))
    y = labels[:n]
    if sum(y) == 0:
        return
    got = _solve(y, [0.5] * n)
    assert math.isclose(got, sum(y) / n, rel_tol=1e-12, abs_tol=1e-12), (
        "with every score tied, AP must equal the positive rate"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 1), min_size=1, max_size=25),
    st.lists(st.integers(-6, 6), min_size=1, max_size=25),
)
def test_affine_change_of_scores_keeps_ap_and_stays_in_unit_interval(labels, grid):
    n = min(len(labels), len(grid))
    y = labels[:n]
    if sum(y) == 0:
        return
    s = [v / 4 for v in grid[:n]]
    base = _solve(y, s)
    shifted = _solve(y, [3 * v + 1 for v in s])
    assert 0.0 <= base <= 1.0, "AP must lie in [0, 1]"
    assert math.isclose(base, shifted, rel_tol=1e-12, abs_tol=1e-12), (
        "AP depends only on the ranking of the scores, so a strictly increasing affine map keeps it"
    )


@pytest.mark.parametrize(
    ("y_true", "scores", "mistake"),
    [
        (
            [0, 0, 0],
            [0.1, 0.5, 0.9],
            "input with no positives was accepted: raise ValueError for it",
        ),
        ([], [], "empty input was accepted: raise ValueError when there are no samples"),
        ([0, 2, 1, 0, 1], [0.1, 0.5, 0.9, 0.2, 0.4], "labels other than 0 and 1 were accepted"),
        ([0, 1, 1], [0.1, float("nan"), 0.9], "NaN scores were accepted: raise ValueError for NaN"),
        ([0, 1, 1], [0.1, 0.9], "unequal lengths were accepted: raise ValueError for them"),
        ([0, 1], [0.1, float("inf")], "infinite scores were accepted: raise ValueError"),
        ([[0, 1]], [[0.1, 0.9]], "matrix input was accepted: both inputs must be one-dimensional"),
    ],
    ids=[
        "no-positives",
        "empty",
        "bad-label",
        "nan-score",
        "length-mismatch",
        "infinite",
        "matrix",
    ],
)
def test_invalid_input_raises_value_error(y_true, scores, mistake):
    _expect_value_error(lambda: average_precision(y_true, scores), mistake)


def test_inputs_are_not_mutated():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.4, 0.9, 0.2, 0.7, 0.5, 0.1])
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
        lambda: average_precision(labels, scores),
        "a two-dimensional label array was accepted or crashed: y_true must be one-dimensional, "
        "so raise ValueError",
    )


PROBLEM = "cml/roc-pr-curves-and-auc/p02-average-precision-step"


@pytest.fixture(scope="module")
def breast_cancer():
    return load("breast_cancer")


def test_matches_scikit_learn_on_raw_breast_cancer_with_one_in_twenty_positives(breast_cancer):
    names = list(breast_cancer.feature_names)
    malignant = 1 - np.asarray(breast_cancer.y).astype(int)  # 1 = malignant, the positive class
    gen = rng(f"{PROBLEM}:breast-cancer-rare:0")
    positives = gen.choice(np.flatnonzero(malignant == 1), size=18, replace=False)
    rows = np.concatenate([positives, np.flatnonzero(malignant == 0)])
    for name in ("worst concave points", "area error"):
        column = breast_cancer.X[rows, names.index(name)]
        expected = average_precision_score(malignant[rows], column)
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
        "AP changed when the scores were scaled by a tiny constant: tied scores are exact "
        "equality, so no absolute tolerance may merge distinct thresholds"
    )


def test_read_only_input_arrays_are_accepted_and_left_alone():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.4, 0.9, 0.2, 0.7, 0.5, 0.1])
    for array in (y, scores):
        array.setflags(write=False)
    try:
        average_precision(y, scores)
    except Exception as exc:
        if "read-only" in str(exc):
            pytest.fail(
                "an input array was written to in place: inputs must not be mutated, so work "
                "on a copy",
                pytrace=False,
            )
        pytest.fail(
            "a valid call failed when its input arrays were read-only: a correct solution "
            "only reads its inputs",
            pytrace=False,
        )
