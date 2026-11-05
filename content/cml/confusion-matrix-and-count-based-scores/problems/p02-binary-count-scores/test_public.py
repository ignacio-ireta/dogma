import math

import pytest
from submission import binary_scores

EXPECTED_KEYS = {"accuracy", "precision", "recall", "specificity", "f1", "mcc"}


def _solve(*counts):
    try:
        s = binary_scores(*counts)
    except Exception as exc:
        pytest.fail(
            f"valid counts raised {type(exc).__name__}: check the input rules and the formulas"
        )
    if set(s) != EXPECTED_KEYS:
        pytest.fail(
            "the result must have exactly the six named keys: "
            "accuracy, precision, recall, specificity, f1, mcc"
        )
    return s


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_example_scores_match_the_hand_calculation():
    s = _solve(3, 1, 2, 4)
    assert math.isclose(s["accuracy"], 0.7, rel_tol=1e-12), "accuracy must be (TP + TN) / N"
    assert math.isclose(s["precision"], 0.75, rel_tol=1e-12), "precision must be TP / (TP + FP)"
    assert math.isclose(s["recall"], 0.6, rel_tol=1e-12), "recall must be TP / (TP + FN)"
    assert math.isclose(s["specificity"], 0.8, rel_tol=1e-12), "specificity must be TN / (TN + FP)"
    assert math.isclose(s["f1"], 2 / 3, rel_tol=1e-12), "F1 must be 2 TP / (2 TP + FP + FN)"
    assert math.isclose(s["mcc"], 10 / math.sqrt(600), rel_tol=1e-12), "MCC formula differs"


def test_all_negative_predictions_on_rare_positives_give_high_accuracy_and_zero_recall():
    s = _solve(0, 0, 10, 990)
    assert math.isclose(s["accuracy"], 0.99, rel_tol=1e-12), "accuracy must count every sample"
    assert s["recall"] == 0.0, "recall must be 0.0 when no actual positive is found"
    assert s["specificity"] == 1.0, "specificity must be 1.0 when every actual negative is cleared"


def test_zero_denominators_return_zero_not_nan():
    s = _solve(0, 0, 0, 5)
    assert s == {
        "accuracy": 1.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 1.0,
        "f1": 0.0,
        "mcc": 0.0,
    }, "with no positives anywhere, undefined ratios must be 0.0 and accuracy must be 1.0"


def test_negative_count_raises_value_error():
    _expect_value_error(
        lambda: binary_scores(3, -1, 2, 4),
        "a negative count was accepted: raise ValueError for counts below zero",
    )
