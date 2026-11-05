"""Hidden tests: scikit-learn oracle for each averaging mode, identities and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from submission import precision_recall_f1

AVERAGES = ("macro", "micro", "weighted")
PART_NAMES = ("precision", "recall", "f1")
PART_MISTAKES = {
    "precision": "precision differs from the reference: per class it is TP / (TP + FP), and 0.0 "
    "when nothing is predicted for that class; the averaging must follow the mode",
    "recall": "recall differs from the reference: per class it is TP / (TP + FN), and 0.0 when "
    "the class has no true samples; the averaging must follow the mode",
    "f1": "F1 differs from the reference: per class it is 2 TP / (2 TP + FP + FN), and macro F1 "
    "is the mean of the per-class F1 values",
}


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _solve(y_true, y_pred, **kwargs):
    """Call the function on input that must be accepted. A crash is a contract mistake."""
    try:
        result = precision_recall_f1(y_true, y_pred, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the averaging mode and the class set"
        )
    if not isinstance(result, tuple) or len(result) != 3:
        pytest.fail("the result must be a (precision, recall, f1) tuple of three values")
    assert all(isinstance(value, float) for value in result), "return Python float scores"
    return tuple(float(v) for v in result)


def _check(got, expected):
    for i, name in enumerate(PART_NAMES):
        if math.isnan(got[i]):
            pytest.fail(f"{name} is NaN: a zero denominator must give 0.0")
        assert math.isclose(got[i], expected[i], rel_tol=1e-9, abs_tol=1e-12), PART_MISTAKES[name]


def _reference(y_true, y_pred, average, pos_label=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
    return float(p), float(r), float(f)


@st.composite
def _label_pairs(draw, max_label=3):
    n = draw(st.integers(min_value=1, max_value=25))
    y_true = draw(st.lists(st.integers(0, max_label), min_size=n, max_size=n))
    y_pred = draw(st.lists(st.integers(0, max_label), min_size=n, max_size=n))
    return y_true, y_pred


@pytest.mark.parametrize(
    ("average", "seed"),
    [(a, s) for a in AVERAGES for s in range(4)],
    ids=[f"{a}-{s}" for a in AVERAGES for s in range(4)],
)
def test_multiclass_modes_match_scikit_learn(average, seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 6))
    n = int(rng.integers(8, 70))
    weights = rng.random(k) ** 2 + 0.05
    y_true = rng.choice(k, size=n, p=weights / weights.sum())
    y_pred = rng.integers(0, k + 1, size=n)  # one extra label: a class that is only predicted
    expected = _reference(y_true, y_pred, average)
    _check(_solve(y_true, y_pred, average=average), expected)


@pytest.mark.parametrize("seed", range(4), ids=[f"binary-{s}" for s in range(4)])
def test_binary_matches_scikit_learn_for_either_positive_label(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(6, 60))
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_true[0], y_true[1] = 0, 1  # both labels present, so the reference is well defined
    for pos in (0, 1):
        expected = _reference(y_true, y_pred, "binary", pos_label=pos)
        got = _solve(y_true, y_pred, average="binary", pos_label=pos)
        _check(got, expected)


def test_binary_with_string_labels_and_a_named_positive_class():
    y_true = ["yes", "no", "yes", "yes", "no", "no", "yes", "no", "yes", "no", "no", "yes"]
    y_pred = ["yes", "yes", "no", "yes", "no", "no", "yes", "no", "yes", "yes", "no", "yes"]
    got = _solve(y_true, y_pred, average="binary", pos_label="no")
    expected = (0.8, 2 / 3, 8 / 11)
    _check(got, expected)


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(_label_pairs())
def test_micro_equals_accuracy_for_one_label_per_sample(pair):
    y_true, y_pred = pair
    accuracy = accuracy_score(y_true, y_pred)
    micro = _solve(y_true, y_pred, average="micro")
    for i, name in enumerate(PART_NAMES):
        assert math.isclose(micro[i], accuracy, rel_tol=1e-12, abs_tol=1e-12), (
            f"micro {name} must equal accuracy for single-label multi-class data: "
            "pool TP, FP and FN over all classes"
        )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(_label_pairs())
def test_weighted_recall_equals_accuracy(pair):
    y_true, y_pred = pair
    accuracy = accuracy_score(y_true, y_pred)
    weighted = _solve(y_true, y_pred, average="weighted")
    assert math.isclose(weighted[1], accuracy, rel_tol=1e-12, abs_tol=1e-12), (
        "weighted recall must equal accuracy: weight each class's recall by its true support"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(_label_pairs())
def test_scores_do_not_depend_on_the_names_of_the_labels(pair):
    y_true, y_pred = pair
    names = {v: f"class-{(7 * v + 3) % 11}" for v in range(4)}
    for average in AVERAGES:
        numeric = _solve(y_true, y_pred, average=average)
        renamed = _solve([names[v] for v in y_true], [names[v] for v in y_pred], average=average)
        for i in range(3):
            assert math.isclose(numeric[i], renamed[i], rel_tol=1e-12, abs_tol=1e-12), (
                "scores changed when the labels were renamed: compare classes by equality only"
            )


def test_macro_f1_is_the_mean_of_per_class_f1_not_the_harmonic_of_the_means():
    got = _solve([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 0, 1, 2, 2, 0], average="macro")
    assert math.isclose(got[2], 179 / 330, rel_tol=1e-12), (
        "macro F1 must be the mean of the per-class F1 values (179/330 for this case)"
    )


def test_predicted_only_class_is_part_of_the_macro_average():
    got = _solve([0, 0, 1, 1], [0, 2, 1, 2], average="macro")
    _check(got, (2 / 3, 1 / 3, 4 / 9))


def test_zero_denominators_for_a_class_give_zero_in_macro():
    got = _solve([0, 1, 2], [0, 1, 1], average="macro")
    _check(got, (0.5, 2 / 3, 5 / 9))


def test_absent_binary_positive_label_returns_three_zeros():
    _check(_solve(["a", "b"], ["b", "a"], pos_label="missing"), (0.0, 0.0, 0.0))


def test_binary_rejects_more_than_two_labels():
    _expect_value_error(
        lambda: precision_recall_f1([0, 1, 2], [0, 1, 2], average="binary"),
        "three labels were accepted by binary averaging: raise ValueError above two labels",
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([], []), ([0, 1], [0])],
    ids=["empty", "length-mismatch"],
)
def test_empty_or_mismatched_input_raises_value_error(y_true, y_pred):
    _expect_value_error(
        lambda: precision_recall_f1(y_true, y_pred, average="macro"),
        "invalid input was accepted: raise ValueError for empty input or unequal lengths",
    )


def test_inputs_are_not_mutated():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 2, 2, 1, 1])
    before_true, before_pred = y_true.copy(), y_pred.copy()
    _solve(y_true, y_pred, average="weighted")
    assert np.array_equal(y_true, before_true), (
        "y_true was changed: compute without modifying inputs"
    )
    assert np.array_equal(y_pred, before_pred), (
        "y_pred was changed: compute without modifying inputs"
    )


def test_f1_is_computed_when_false_positives_are_twice_the_true_positives():
    got = _solve([1, 0, 0], [1, 1, 1], average="binary")
    assert math.isclose(got[2], 0.5, rel_tol=1e-12, abs_tol=1e-12), (
        "F1 is 2 TP / (2 TP + FP + FN) and must not be zeroed when 2 TP - FP + FN is zero"
    )
