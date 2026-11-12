"""Hidden tests: a brute-force search over thresholds, the tie rules, and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import math
import time
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.metrics import f1_score
from submission import best_f1_threshold

from dogma_data import load, rng, sample_rows

THRESHOLD_MISTAKE = (
    "threshold differs from the brute-force search: a sample is positive when its score is at "
    "least the threshold, candidates are the distinct scores, and equal F1 goes to the largest"
)
F1_MISTAKE = "F1 differs from the brute-force search: F1 = 2 TP / (2 TP + FP + FN) at the threshold"


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


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


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
        result = best_f1_threshold(y_true, scores)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the sweep over the distinct scores"
        )
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("the result must be a (threshold, f1) pair")
    return float(result[0]), float(result[1])


def _brute_force(y_true, scores):
    """Try every distinct score as the threshold. Ascending order with >= keeps the largest tie."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    best_threshold, best_f1 = None, -1.0
    for t in sorted(set(s.tolist())):
        pred = s >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 >= best_f1:
            best_threshold, best_f1 = t, f1
    return best_threshold, best_f1


@pytest.mark.parametrize("seed", range(6), ids=[f"case-{s}" for s in range(6)])
def test_matches_brute_force_on_tied_and_continuous_scores(seed):
    # Real diagnoses scored by one real breast_cancer feature. Even seeds use a randomly chosen
    # feature as measured. Odd seeds use a weak predictor (mean fractal dimension, or the
    # texture, smoothness or symmetry error) rounded to whole standardised units, so many
    # tumours share a score and tied tumours differ in diagnosis.
    gen = rng(f"best-f1-p02:brute-force:{seed}")
    n = int(gen.integers(3, 40))
    features, target = sample_rows("breast_cancer", gen, n, standardize=True)
    y = target.astype(np.int64)
    if seed % 2:
        scores = np.round(features[:, int(gen.choice([9, 11, 14, 18]))])
    else:
        scores = features[:, int(gen.integers(features.shape[1]))]
    ref_threshold, ref_f1 = _brute_force(y, scores)
    threshold, f1 = _solve(y, scores)
    assert threshold == ref_threshold, THRESHOLD_MISTAKE
    assert math.isclose(f1, ref_f1, rel_tol=1e-12, abs_tol=1e-12), F1_MISTAKE


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(0, 1), min_size=2, max_size=20),
    st.lists(st.integers(0, 4), min_size=2, max_size=20),
)
def test_returned_f1_is_the_scikit_learn_maximum_over_all_thresholds(labels, grid):
    n = min(len(labels), len(grid))
    y = labels[:n]
    scores = [v / 4 for v in grid[:n]]
    if sum(y) == 0:
        return  # the no-positives rule is tested separately
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_elsewhere = max(
            f1_score(y, [s >= t for s in scores], zero_division=0) for t in set(scores)
        )
    threshold, f1 = _solve(y, scores)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at_threshold = f1_score(y, [s >= threshold for s in scores], zero_division=0)
    assert math.isclose(f1, at_threshold, rel_tol=1e-12, abs_tol=1e-12), F1_MISTAKE
    assert math.isclose(f1, best_elsewhere, rel_tol=1e-12, abs_tol=1e-12), (
        "a different threshold has a higher F1: sweep every distinct score, not a subset"
    )


def test_tied_scores_are_never_split():
    threshold, f1 = _solve([1, 0], [0.5, 0.5])
    assert threshold == 0.5 and math.isclose(f1, 2 / 3, rel_tol=1e-12), (
        "tied scores were split: samples with equal scores must be flagged together"
    )


def test_the_largest_threshold_wins_an_exact_tie_in_f1():
    threshold, f1 = _solve([0, 0, 1, 1], [0.0, 0.0, 0.0, 1.0])
    assert threshold == 1.0, "equal best F1 at two thresholds: return the largest threshold"
    assert math.isclose(f1, 2 / 3, rel_tol=1e-12), F1_MISTAKE


def test_a_score_equal_to_the_threshold_counts_as_positive():
    threshold, f1 = _solve([0, 1], [0.2, 0.8])
    assert (threshold, f1) == (0.8, 1.0), (
        "a sample whose score equals the threshold must be predicted positive (score >= threshold)"
    )


def test_flagging_everything_is_a_candidate():
    threshold, f1 = _solve([1, 1, 1], [0.3, 0.9, 0.5])
    assert (threshold, f1) == (0.3, 1.0), (
        "the lowest score must be a candidate threshold, which flags every sample"
    )


def test_no_positives_return_the_highest_score_with_zero_f1():
    threshold, f1 = _solve([0, 0, 0, 0], [0.2, 0.9, 0.9, 0.1])
    assert (threshold, f1) == (0.9, 0.0), "with no positives, return (highest score, 0.0)"


def test_result_is_one_of_the_scores_not_a_position():
    scores = [0.15, 0.6, 0.6, 0.95, 0.3]
    threshold, _ = _solve([0, 1, 0, 1, 1], scores)
    assert threshold in scores, "the threshold must be one of the score values, not an index"


def test_large_input_is_swept_quickly_and_consistently():
    rng = np.random.default_rng(3)
    n = 200_000
    y = _real_ints(rng, 0, 2, n)
    scores = np.round(_real_unit(rng, n) * 1000) / 1000.0
    start = time.perf_counter()
    threshold, f1 = _solve(y, scores)
    elapsed = time.perf_counter() - start
    assert elapsed < 10, (
        "the sweep is too slow for 200,000 scores: sort once and use cumulative counts"
    )
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    assert math.isclose(f1, 2 * tp / (2 * tp + fp + fn), rel_tol=1e-12), (
        F1_MISTAKE + " (at the returned threshold, on the large input)"
    )


@pytest.mark.parametrize(
    ("y_true", "scores", "mistake"),
    [
        ([0, 1], [0.1, float("nan")], "NaN scores were accepted: raise ValueError for NaN scores"),
        ([0, 1], [0.1, float("inf")], "infinite scores were accepted: raise ValueError for them"),
        ([0, 2], [0.1, 0.9], "labels other than 0 and 1 were accepted: raise ValueError for them"),
        ([0, 1, 1], [0.1, 0.9], "unequal lengths were accepted: raise ValueError for them"),
        ([], [], "empty input was accepted: raise ValueError when there are no samples"),
        ([[0, 1]], [0.1, 0.9], "two-dimensional labels were accepted: raise ValueError"),
    ],
    ids=["nan-score", "inf-score", "bad-label", "length-mismatch", "empty", "two-dimensional"],
)
def test_invalid_input_raises_value_error(y_true, scores, mistake):
    _expect_value_error(lambda: best_f1_threshold(y_true, scores), mistake)


def test_inputs_are_not_mutated():
    y = np.array([1, 0, 1, 1, 0, 0])
    scores = np.array([0.4, 0.2, 0.9, 0.9, 0.1, 0.6])
    before_y, before_scores = y.copy(), scores.copy()
    _solve(y, scores)
    assert np.array_equal(y, before_y), "y_true was reordered or changed: work on sorted copies"
    assert np.array_equal(scores, before_scores), (
        "scores were reordered or changed: sort a copy, never the caller's array"
    )


def test_two_dimensional_labels_with_one_dimensional_scores_raise_value_error():
    _expect_value_error(
        lambda: best_f1_threshold([[0, 1], [1, 0]], [0.1, 0.9]),
        "two-dimensional labels were accepted: y_true must be one-dimensional, so raise ValueError",
    )


def test_one_dimensional_labels_with_two_dimensional_scores_raise_value_error():
    _expect_value_error(
        lambda: best_f1_threshold([0, 1], [[0.1, 0.9], [0.4, 0.2]]),
        "two-dimensional scores were accepted: scores must be one-dimensional, so raise ValueError",
    )


PROBLEM = "cml/precision-recall-and-f1/p02-best-f1-threshold"


@pytest.fixture(scope="module")
def breast_cancer():
    return load("breast_cancer")


def test_best_f1_matches_scikit_learn_sweep_on_raw_breast_cancer_with_rare_positives(breast_cancer):
    names = list(breast_cancer.feature_names)
    malignant = 1 - np.asarray(breast_cancer.y).astype(int)  # 1 = malignant, the positive class
    gen = rng(f"{PROBLEM}:breast-cancer-rare:0")
    positives = gen.choice(np.flatnonzero(malignant == 1), size=18, replace=False)
    rows = np.concatenate([positives, np.flatnonzero(malignant == 0)])
    y = malignant[rows]
    for name in ("worst concave points", "mean texture"):
        column = breast_cancer.X[rows, names.index(name)]
        candidates = np.unique(column)
        f1s = np.array([f1_score(y, column >= t, zero_division=0) for t in candidates])
        expected_threshold = candidates[np.flatnonzero(f1s == f1s.max())[-1]]
        threshold, f1 = _solve(y, column)
        assert math.isclose(f1, f1s.max(), rel_tol=1e-9, abs_tol=1e-12), F1_MISTAKE
        assert threshold == expected_threshold, THRESHOLD_MISTAKE


def test_only_the_order_and_equality_of_scores_matter_at_tiny_magnitudes(breast_cancer):
    names = list(breast_cancer.feature_names)
    malignant = 1 - np.asarray(breast_cancer.y).astype(int)
    column = breast_cancer.X[:, names.index("worst concave points")]
    threshold, f1 = _solve(malignant, column)
    tiny_threshold, tiny_f1 = _solve(malignant, column * 1e-9)
    assert math.isclose(f1, tiny_f1, rel_tol=1e-12, abs_tol=1e-12), (
        "best F1 changed when the scores were scaled by a tiny constant: tied scores are exact "
        "equality, so no absolute tolerance may merge distinct thresholds"
    )
    assert math.isclose(tiny_threshold / 1e-9, threshold, rel_tol=1e-9), (
        "the threshold must move with the scores: it is one of the score values"
    )


def test_boolean_labels_are_valid_and_match_integer_labels():
    threshold, f1 = _solve([False, False, True, True], [0.1, 0.4, 0.35, 0.8])
    assert threshold == 0.35 and math.isclose(f1, 0.8, rel_tol=1e-12), (
        "boolean labels were rejected or mishandled: False and True are valid and must behave "
        "like 0 and 1"
    )
