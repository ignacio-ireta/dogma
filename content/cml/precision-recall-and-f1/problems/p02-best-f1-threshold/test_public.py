import math

import pytest
from submission import best_f1_threshold


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_example_picks_the_threshold_with_the_best_f1():
    threshold, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert threshold == 0.35, "the best threshold is the one that flags TP = 2, FP = 1, FN = 0"
    assert math.isclose(f1, 0.8, rel_tol=1e-12), (
        "F1 at that threshold must be 2 TP / (2 TP + FP + FN)"
    )


def test_tied_scores_are_flagged_together_and_the_largest_threshold_wins():
    threshold, f1 = best_f1_threshold([0, 0, 1, 1], [0.0, 0.0, 0.0, 1.0])
    assert threshold == 1.0, "two thresholds reach the same F1: return the largest one"
    assert math.isclose(f1, 2 / 3, rel_tol=1e-12), "F1 must be 2 / 3 for this case"


def test_no_positives_return_the_highest_score_and_zero_f1():
    threshold, f1 = best_f1_threshold([0, 0, 0], [0.1, 0.7, 0.4])
    assert (threshold, f1) == (0.7, 0.0), "with no positives, return (highest score, 0.0)"


def test_scores_that_are_not_finite_raise_value_error():
    _expect_value_error(
        lambda: best_f1_threshold([0, 1], [0.1, float("nan")]),
        "NaN scores were accepted: raise ValueError for scores that are not finite",
    )
