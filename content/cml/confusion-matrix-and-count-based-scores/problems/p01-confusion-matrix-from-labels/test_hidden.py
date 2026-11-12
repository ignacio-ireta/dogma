"""Hidden tests: an oracle comparison with scikit-learn, property checks and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from submission import confusion_matrix

from dogma_data import load, rng, sample_rows

MATRIX_MISTAKE = (
    "matrix differs from the reference: rows are true labels, columns are predicted labels, "
    "and each sample adds one to its (true, predicted) cell"
)
LABELS_MISTAKE = (
    "matrix with explicit labels differs: keep the order of labels, ignore samples whose labels "
    "are not listed, and give a listed label that never occurs a zero row and column"
)


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


PROBLEM = "cml/confusion-matrix-and-count-based-scores/p01-confusion-matrix-from-labels"


@pytest.mark.parametrize(
    "seed", [0, 1, 2, 3, 4], ids=["case-0", "case-1", "case-2", "case-3", "case-4"]
)
def test_matches_scikit_learn_on_random_integer_labels(seed):
    generator = rng(f"{PROBLEM}:random-labels:{seed}")
    n = int(generator.integers(5, 80))
    k = int(generator.integers(2, 6))
    y_true = _real_ints(generator, 0, k, n)
    y_pred = _real_ints(generator, 0, k, n)
    expected = sklearn_confusion_matrix(y_true, y_pred)
    got = _solve(y_true, y_pred)
    assert got.shape == expected.shape, (
        "matrix shape differs: the default label set is the sorted union of "
        "true and predicted labels"
    )
    np.testing.assert_array_equal(got, expected, err_msg=MATRIX_MISTAKE)


@pytest.mark.parametrize("seed", [10, 11, 12], ids=["subset-0", "subset-1", "subset-2"])
def test_explicit_labels_match_scikit_learn_with_unseen_and_dropped_labels(seed):
    generator = rng(f"{PROBLEM}:explicit-labels:{seed}")
    n = int(generator.integers(10, 60))
    y_true = _real_ints(generator, 0, 5, n)
    y_pred = _real_ints(generator, 0, 5, n)
    labels = [int(v) for v in generator.permutation(7)[:5]]
    expected = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    got = _solve(y_true, y_pred, labels=labels)
    np.testing.assert_array_equal(got, expected, err_msg=LABELS_MISTAKE)


@pytest.fixture(scope="module")
def breast_cancer():
    return load("breast_cancer")


def test_matches_scikit_learn_on_breast_cancer_feature_scores(breast_cancer):
    malignant = (np.asarray(breast_cancer.y) == 0).astype(int)
    feature = list(breast_cancer.feature_names).index("worst concave points")
    scores = breast_cancer.X[:, feature]
    predictions = (scores >= 0.04).astype(int)

    expected = sklearn_confusion_matrix(malignant, predictions, labels=[0, 1])
    got = _solve(malignant, predictions)
    assert got.shape == expected.shape, (
        "matrix shape differs: the default label set is the sorted union of "
        "true and predicted labels"
    )
    np.testing.assert_array_equal(got, expected, err_msg=MATRIX_MISTAKE)

    labels = [1, 2]  # omit benign label 0 and include unseen label 2
    expected_subset = sklearn_confusion_matrix(malignant, predictions, labels=labels)
    got_subset = _solve(malignant, predictions, labels=labels)
    np.testing.assert_array_equal(got_subset, expected_subset, err_msg=LABELS_MISTAKE)


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


def test_unsorted_array_inputs_are_not_reordered():
    y_true = np.array([2, 0, 1, 2])
    y_pred = np.array([1, 2, 0, 2])
    before_true, before_pred = y_true.copy(), y_pred.copy()
    _solve(y_true, y_pred)
    np.testing.assert_array_equal(
        y_true,
        before_true,
        err_msg="y_true was reordered: count without changing the caller's arrays",
    )
    np.testing.assert_array_equal(
        y_pred,
        before_pred,
        err_msg="y_pred was reordered: count without changing the caller's arrays",
    )


def test_integer_labels_sort_numerically_for_the_default_label_set():
    # Labels sort numerically to 2, 10 (not as text, where "10" comes before "2").
    # Row 2: one sample predicted 2 (column 0), one predicted 10 (column 1).
    # Row 10: one sample predicted 10 (column 1).
    got = _solve([2, 2, 10], [2, 10, 10])
    expected = np.array([[1, 1], [0, 1]])
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="integer labels must sort as numbers, not as text",
    )
