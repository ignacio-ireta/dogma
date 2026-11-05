# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import numpy as np
import pytest
from submission import prediction_sets

CAL_PROBS = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.5, 0.4, 0.1]])
CAL_LABELS = np.array([0, 1, 2, 0])


def test_confident_row_keeps_only_the_class_under_the_threshold():
    mask = prediction_sets(CAL_PROBS, CAL_LABELS, np.array([[0.7, 0.2, 0.1]]), 0.6)
    np.testing.assert_array_equal(
        mask,
        [[True, False, False]],
        err_msg="q = 0.4 (2nd smallest score): only class 0, with score 0.3, is inside",
    )


def test_threshold_equality_is_inside_and_all_false_row_is_empty():
    mask = prediction_sets(
        CAL_PROBS, CAL_LABELS, np.array([[0.45, 0.45, 0.1], [0.5, 0.4, 0.1]]), 0.2
    )
    np.testing.assert_array_equal(
        mask,
        [[False, False, False], [True, False, False]],
        err_msg="q = 0.5: a score equal to q is inside; a row with no score under q is empty",
    )


def test_invalid_row_sum_raises_value_error():
    with pytest.raises(ValueError):
        prediction_sets(CAL_PROBS, CAL_LABELS, np.array([[0.5, 0.4, 0.2]]), 0.2)
