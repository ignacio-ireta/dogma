import math

import pytest
from submission import precision_recall_f1

Y_TRUE = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
Y_PRED = [0, 0, 0, 1, 1, 0, 1, 2, 2, 0]


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_binary_scores_the_positive_class_from_its_counts():
    got = precision_recall_f1(
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    )
    expected = (5 / 7, 5 / 6, 10 / 13)
    assert all(math.isclose(g, e, rel_tol=1e-12) for g, e in zip(got, expected, strict=True)), (
        "binary precision, recall or F1 differs: use the counts of the positive class only"
    )


def test_macro_is_the_unweighted_mean_of_per_class_scores():
    got = precision_recall_f1(Y_TRUE, Y_PRED, average="macro")
    expected = (49 / 90, 5 / 9, 179 / 330)
    assert all(math.isclose(g, e, rel_tol=1e-12) for g, e in zip(got, expected, strict=True)), (
        "macro scores differ: average the per-class precision, recall and F1 with equal weights"
    )


def test_micro_equals_accuracy_for_single_label_multiclass():
    got = precision_recall_f1(Y_TRUE, Y_PRED, average="micro")
    assert all(math.isclose(g, 0.6, rel_tol=1e-12) for g in got), (
        "micro scores must pool the counts over classes; "
        "for one label per sample they equal accuracy"
    )


def test_zero_predictions_give_zero_precision_not_nan():
    got = precision_recall_f1([1, 0], [0, 0])
    assert got == (0.0, 0.0, 0.0), (
        "with no predicted positives, precision, recall and F1 must be 0.0"
    )


def test_unknown_average_raises_value_error():
    _expect_value_error(
        lambda: precision_recall_f1(Y_TRUE, Y_PRED, average="samples"),
        "an unknown average was accepted: raise ValueError for modes other than the four named",
    )
