"""Hidden tests: an oracle comparison with scikit-learn, property checks and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from submission import confusion_matrix

MATRIX_MISTAKE = (
    "matrix differs from the reference: rows are true labels, columns are predicted labels, "
    "and each sample adds one to its (true, predicted) cell"
)
LABELS_MISTAKE = (
    "matrix with explicit labels differs: keep the order of labels, ignore samples whose labels "
    "are not listed, and give a listed label that never occurs a zero row and column"
)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _solve(y_true, y_pred, labels=None):
    """Call the function on valid input. A crash is a contract mistake."""
    try:
        result = confusion_matrix(y_true, y_pred, labels=labels)
    except Exception as exc:
        pytest.fail(
            f"a valid input raised {type(exc).__name__}: check the label set and the counting loop"
        )
    assert isinstance(result, np.ndarray), "return a NumPy array, not a nested sequence"
    return result


@st.composite
def _label_pairs(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    y_true = draw(st.lists(st.integers(0, 4), min_size=n, max_size=n))
    y_pred = draw(st.lists(st.integers(0, 4), min_size=n, max_size=n))
    return y_true, y_pred


@pytest.mark.parametrize(
    "seed", [0, 1, 2, 3, 4], ids=["case-0", "case-1", "case-2", "case-3", "case-4"]
)
def test_matches_scikit_learn_on_random_integer_labels(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    k = int(rng.integers(2, 6))
    y_true = rng.integers(0, k, size=n)
    y_pred = rng.integers(0, k, size=n)
    expected = sklearn_confusion_matrix(y_true, y_pred)
    got = _solve(y_true, y_pred)
    assert got.shape == expected.shape, (
        "matrix shape differs: the default label set is the sorted union of y_true and y_pred"
    )
    np.testing.assert_array_equal(got, expected, err_msg=MATRIX_MISTAKE)


@pytest.mark.parametrize("seed", [10, 11, 12], ids=["subset-0", "subset-1", "subset-2"])
def test_explicit_labels_match_scikit_learn_with_unseen_and_dropped_labels(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 60))
    y_true = rng.integers(0, 5, size=n)
    y_pred = rng.integers(0, 5, size=n)
    labels = [int(v) for v in rng.permutation(7)[:5]]
    expected = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    got = _solve(y_true, y_pred, labels=labels)
    np.testing.assert_array_equal(got, expected, err_msg=LABELS_MISTAKE)


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(_label_pairs())
def test_row_and_column_sums_are_the_true_and_predicted_label_counts(pair):
    y_true, y_pred = pair
    labels = sorted(set(y_true) | set(y_pred))
    matrix = _solve(y_true, y_pred)
    assert matrix.shape == (len(labels), len(labels)), (
        "matrix shape differs: one row and one column per label in y_true or y_pred"
    )
    assert list(matrix.sum(axis=1)) == [y_true.count(v) for v in labels], (
        "row sums must count the true labels: rows index y_true"
    )
    assert list(matrix.sum(axis=0)) == [y_pred.count(v) for v in labels], (
        "column sums must count the predicted labels: columns index y_pred"
    )


def test_rows_are_true_labels_on_asymmetric_data():
    got = _solve([0, 0, 0, 0, 1], [0, 1, 1, 1, 1])
    expected = np.array([[1, 3], [0, 1]])
    np.testing.assert_array_equal(got, expected, err_msg=MATRIX_MISTAKE)


def test_label_that_is_only_predicted_still_gets_a_column():
    got = _solve([0, 0], [0, 7])
    expected = np.array([[1, 1], [0, 0]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="a label that only appears in y_pred must get a column: the label set is the union",
    )


def test_listed_order_sets_rows_and_columns():
    got = _solve([0, 0, 2], [0, 2, 2], labels=[2, 0])
    expected = np.array([[1, 0], [1, 1]])
    np.testing.assert_array_equal(got, expected, err_msg=LABELS_MISTAKE)


def test_samples_with_a_label_outside_labels_are_ignored_not_rejected():
    got = _solve([0, 1, 2, 2], [0, 2, 2, 1], labels=[0, 1])
    expected = np.array([[1, 0], [0, 0]])
    np.testing.assert_array_equal(got, expected, err_msg=LABELS_MISTAKE)


def test_repeated_pairs_accumulate_their_counts():
    got = _solve([1, 1, 1], [0, 0, 0])
    expected = np.array([[0, 0], [3, 0]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="a repeated (true, predicted) pair must add one per sample, not overwrite its cell",
    )


def test_duplicate_labels_raise_value_error():
    _expect_value_error(
        lambda: confusion_matrix([0, 1], [0, 1], labels=[1, 1]),
        "duplicate labels were accepted: raise ValueError when labels repeats a label",
    )


def test_empty_input_raises_value_error():
    _expect_value_error(
        lambda: confusion_matrix([], []),
        "empty input was accepted: raise ValueError when there are no samples",
    )


def test_mismatched_lengths_raise_value_error():
    _expect_value_error(
        lambda: confusion_matrix([0, 1], [0]),
        "mismatched lengths were accepted: raise ValueError when y_true and y_pred differ",
    )


def test_result_is_an_integer_matrix_of_the_right_shape():
    got = _solve([0, 1, 1], [1, 1, 0])
    assert np.issubdtype(got.dtype, np.integer), "counts must be an integer array, not floats"
    assert got.shape == (2, 2), "matrix shape differs: one row and one column per label"


def test_single_label_gives_a_one_by_one_matrix():
    got = _solve([3, 3, 3], [3, 3, 3])
    np.testing.assert_array_equal(
        got, np.array([[3]]), err_msg="one label must give a 1 x 1 matrix of the sample count"
    )


def test_explicit_empty_label_set_returns_empty_integer_matrix():
    got = _solve([0, 1], [1, 0], labels=[])
    assert got.shape == (0, 0), "an empty explicit label set must give a zero-by-zero matrix"
    assert np.issubdtype(got.dtype, np.integer), "empty counts must retain integer dtype"


def test_string_labels_use_python_sort_order_for_the_default_label_set():
    got = _solve(["b", "B", "a"], ["a", "B", "b"])
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="string labels must use Python's sort order: 'B' sorts before 'a' and 'b'",
    )


def test_inputs_and_labels_list_are_not_mutated():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([2, 1, 0, 2])
    labels = [2, 1, 0]
    before_true, before_pred = y_true.copy(), y_pred.copy()
    _solve(y_true, y_pred, labels=labels)
    assert np.array_equal(y_true, before_true), "y_true was changed: count without modifying inputs"
    assert np.array_equal(y_pred, before_pred), "y_pred was changed: count without modifying inputs"
    assert labels == [2, 1, 0], "the labels list was changed: keep the caller's order"


def test_default_label_order_is_sorted_not_set_order():
    got = _solve([9], [1])
    expected = np.array([[0, 0], [1, 0]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="with labels=None the labels must be in sorted order, not in the order of a set",
    )
