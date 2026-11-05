"""Hidden tests: scikit-learn oracle on rebuilt labels, exact large-count MCC, identities, edges.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from submission import binary_scores

KEY_MISTAKES = {
    "accuracy": "accuracy differs: it is (TP + TN) / N, the share of all samples that are right",
    "precision": (
        "precision differs: it is TP / (TP + FP), and 0.0 when nothing is predicted positive"
    ),
    "recall": "recall differs: it is TP / (TP + FN), and 0.0 when there are no actual positives",
    "specificity": "specificity differs: it is TN / (TN + FP), the true negative rate",
    "f1": "F1 differs: it is 2 TP / (2 TP + FP + FN), and 0.0 when that denominator is 0",
    "mcc": (
        "MCC differs: it is the signed correlation of the 2 x 2 table, and 0.0 when a marginal is 0"
    ),
}
EXPECTED_KEYS = {"accuracy", "precision", "recall", "specificity", "f1", "mcc"}


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _solve(tp, fp, fn, tn):
    """Call the function on valid counts. A crash is a contract mistake."""
    try:
        result = binary_scores(tp, fp, fn, tn)
    except Exception as exc:
        pytest.fail(
            f"valid counts raised {type(exc).__name__}: check the input rules, the zero-division "
            "rule and the MCC formula"
        )
    if set(result) != EXPECTED_KEYS:
        pytest.fail(
            "the result must have exactly the six named keys: accuracy, precision, recall, "
            "specificity, f1, mcc"
        )
    return result


def _labels_from_counts(tp, fp, fn, tn):
    """Rebuild label vectors whose 2 x 2 table is exactly (tp, fp, fn, tn)."""
    y_true = [1] * (tp + fn) + [0] * (fp + tn)
    y_pred = [1] * tp + [0] * fn + [1] * fp + [0] * tn
    return np.array(y_true), np.array(y_pred)


def _reference(tp, fp, fn, tn):
    """scikit-learn with the same zero-division convention. Degenerate MCC warnings are expected."""
    y_true, y_pred = _labels_from_counts(tp, fp, fn, tn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0),
            "specificity": recall_score(
                y_true, y_pred, labels=[0, 1], pos_label=0, zero_division=0
            ),
            "f1": f1_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }


def _exact_reference(tp, fp, fn, tn):
    """Independent route: exact rational arithmetic, and the MCC with Python's exact integers."""
    total = tp + fp + fn + tn

    def ratio(num, den):
        return float(Fraction(num, den)) if den else 0.0

    denominator_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator_sq) if denominator_sq else 0.0
    return {
        "accuracy": ratio(tp + tn, total),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": mcc,
    }


@pytest.mark.parametrize(
    "counts",
    [(3, 1, 2, 4), (7, 0, 0, 9), (0, 5, 5, 0), (1, 1, 1, 1), (12, 3, 4, 20), (2, 9, 1, 3)],
    ids=["mixed", "perfect", "fully-wrong", "uniform", "moderate", "skewed"],
)
def test_matches_scikit_learn_with_the_zero_division_convention(counts):
    expected = _reference(*counts)
    got = _solve(*counts)
    for key in sorted(EXPECTED_KEYS):
        assert math.isclose(got[key], expected[key], rel_tol=1e-9, abs_tol=1e-12), KEY_MISTAKES[key]


@pytest.mark.parametrize(
    "counts",
    [
        (3_000_000, 1_500_000, 200_000, 4_000_000),
        (950_000, 40_000, 60_000, 8_950_000),
        (2_000_001, 3, 999_999, 2_000_000),
    ],
    ids=["large-balanced", "large-rare-positives", "large-odd"],
)
def test_large_counts_match_exact_integer_arithmetic(counts):
    expected = _exact_reference(*counts)
    got = _solve(*counts)
    for key in sorted(EXPECTED_KEYS):
        assert math.isclose(got[key], expected[key], rel_tol=1e-9, abs_tol=1e-12), (
            f"{KEY_MISTAKES[key]} (large counts: intermediate products may overflow a 64-bit "
            "integer)"
        )


def test_seeded_random_counts_match_scikit_learn():
    rng = np.random.default_rng(2901)
    for _ in range(25):
        counts = [int(value) for value in rng.integers(0, 30, size=4)]
        counts[0] += 1
        expected = _reference(*counts)
        got = _solve(*counts)
        for key in sorted(EXPECTED_KEYS):
            message = KEY_MISTAKES[key]
            assert math.isclose(got[key], expected[key], rel_tol=1e-9, abs_tol=1e-12), message


def test_specificity_is_not_the_negative_predictive_value():
    got = _solve(2, 5, 1, 30)
    assert math.isclose(got["specificity"], 30 / 35, rel_tol=1e-12), KEY_MISTAKES["specificity"]


def test_accuracy_is_not_balanced_accuracy():
    got = _solve(1, 0, 9, 990)
    assert math.isclose(got["accuracy"], 991 / 1000, rel_tol=1e-12), (
        "accuracy differs: it is the share of all samples right, not the average of recall and "
        "specificity"
    )


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((0, 0, 4, 6), {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0}),
        ((0, 3, 0, 7), {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0}),
        ((5, 0, 2, 0), {"specificity": 0.0, "mcc": 0.0}),
        ((0, 0, 0, 9), {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0}),
    ],
    ids=[
        "nothing-predicted-positive",
        "no-actual-positives",
        "no-actual-negatives",
        "all-negative",
    ],
)
def test_undefined_ratios_are_zero_not_nan(counts, expected):
    got = _solve(*counts)
    for key, value in expected.items():
        assert not math.isnan(got[key]), f"{key} is NaN: {KEY_MISTAKES[key]}"
        assert got[key] == value, KEY_MISTAKES[key]


def test_returns_exactly_the_six_float_scores():
    got = _solve(4, 2, 3, 5)
    assert set(got) == EXPECTED_KEYS, "the result must have exactly the six named keys"
    assert all(isinstance(v, float) for v in got.values()), "every score must be a float"


def test_numpy_integer_counts_are_accepted():
    try:
        got = binary_scores(np.int64(3), np.int64(1), np.int64(2), np.int64(4))
    except Exception as exc:
        pytest.fail(
            f"NumPy integer counts were rejected or crashed with {type(exc).__name__}: "
            "accept any integer type, not only Python int"
        )
    assert math.isclose(got["precision"], 0.75, rel_tol=1e-12), (
        "NumPy integer counts must be scored like Python ints"
    )


@pytest.mark.parametrize(
    "counts",
    [(1, 2.5, 0, 0), (0, 0, 0, 0), (-1, 2, 0, 0)],
    ids=["non-integer", "all-zero", "negative"],
)
def test_invalid_counts_raise_value_error(counts):
    _expect_value_error(
        lambda: binary_scores(*counts),
        "invalid counts were accepted: raise ValueError for a negative or non-integer count, or "
        "when all counts are 0",
    )


def test_whole_valued_float_is_not_a_count_and_raises_value_error():
    _expect_value_error(
        lambda: binary_scores(2.0, 0, 1, 3),
        "a float count was accepted: counts must be Python or NumPy integers, so a whole-valued "
        "float raises too",
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.integers(0, 60), st.integers(0, 60), st.integers(0, 60), st.integers(0, 60))
def test_swapping_the_two_classes_keeps_accuracy_and_mcc(tp, fp, fn, tn):
    if tp + fp + fn + tn == 0:
        return
    a = _solve(tp, fp, fn, tn)
    b = _solve(tn, fn, fp, tp)
    assert math.isclose(a["accuracy"], b["accuracy"], rel_tol=1e-12), (
        "accuracy changed when the classes were swapped: it must be symmetric"
    )
    assert math.isclose(a["mcc"], b["mcc"], rel_tol=1e-9, abs_tol=1e-12), (
        "MCC changed when the classes were swapped: it must be symmetric"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 60), st.integers(0, 60), st.integers(0, 60), st.integers(0, 60))
def test_f1_is_the_harmonic_mean_of_precision_and_recall(tp, fp, fn, tn):
    got = _solve(tp, fp, fn, tn)
    p, r = got["precision"], got["recall"]
    assert p + r > 0, (
        "precision and recall are both 0 with a positive count: check the denominators"
    )
    assert math.isclose(got["f1"], 2 * p * r / (p + r), rel_tol=1e-12, abs_tol=1e-12), (
        "F1 must be the harmonic mean of precision and recall, not their arithmetic mean"
    )
