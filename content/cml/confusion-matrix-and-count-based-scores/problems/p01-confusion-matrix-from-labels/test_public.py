import numpy as np
import pytest
from submission import confusion_matrix


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_integer_labels_count_true_rows_and_predicted_columns():
    got = confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1])
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="matrix differs: rows must be the true labels and columns the predicted labels",
    )


def test_string_labels_are_sorted_before_they_index_the_matrix():
    got = confusion_matrix(["cat", "dog", "cat"], ["dog", "dog", "cat"])
    expected = np.array([[1, 1], [0, 1]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="string labels must sort (cat before dog) to set the row and column order",
    )


def test_listed_label_that_never_occurs_gets_a_zero_row_and_column():
    got = confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1], labels=[2, 0, 5])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg=(
            "with labels given, use that order, skip samples outside it, "
            "and zero-fill unseen labels"
        ),
    )


def test_length_mismatch_raises_value_error():
    _expect_value_error(
        lambda: confusion_matrix([0, 1], [0, 1, 1]),
        "mismatched lengths were accepted: raise ValueError when y_true and y_pred differ",
    )
