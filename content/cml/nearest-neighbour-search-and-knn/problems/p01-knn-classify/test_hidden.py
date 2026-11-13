import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.neighbors import KNeighborsClassifier
from submission import knn_classify

from dogma_data import rng as data_rng
from dogma_data import sample_rows

RATIO_COLUMNS = (2, 3, 5)  # rooms, bedrooms and occupants per household


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


def _houses(gen, n, d):
    """n real california_housing block groups: d standardised averages and the house value.

    The features are ratios of counts, so no two distances tie exactly, and they are
    heavy-tailed. The value is the median house value in units of 100,000 dollars.
    """
    X, value = sample_rows("california_housing", gen, n, standardize=True)
    return X[:, RATIO_COLUMNS[:d]], value


def _value_band(value, lowest=0):
    """House-value band as an integer label: four bands split at 1.2, 1.8 and 2.6."""
    return np.digitize(value, [1.2, 1.8, 2.6]) + lowest


def _call(X, y, Q, k):
    """Run with NumPy warnings as errors; a crash on valid input is a contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = knn_classify(X, y, Q, k)
    except RuntimeWarning:
        pytest.fail("the distance or vote arithmetic emitted a NumPy warning")
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    out = np.asarray(out)
    if out.shape != (len(Q),):
        pytest.fail("shape mismatch: return one label per query row, as a 1-D array")
    return out


def _expect_value_error(mistake, X, y, Q, k):
    try:
        knn_classify(X, y, Q, k)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a result instead of raising ValueError")


def _reference(X, y, Q, k):
    """Brute force with plain Python: sort by (distance, index), vote, break ties by nearest."""
    out = []
    for q in Q:
        dist = [math.sqrt(sum((a - b) ** 2 for a, b in zip(x, q, strict=True))) for x in X]
        order = sorted(range(len(X)), key=lambda i: (dist[i], i))[:k]
        counts = {}
        for i in order:
            counts[y[i]] = counts.get(y[i], 0) + 1
        top = max(counts.values())
        tied = {label for label, c in counts.items() if c == top}
        for i in order:
            if y[i] in tied:
                out.append(y[i])
                break
    return np.array(out, dtype=np.int64)


def _ties_are_exact(X, q):
    """True when every pair of rows at the same distance from q has the same differences.

    Rows whose differences from the query agree in absolute value, coordinate by coordinate,
    are at exactly the same distance whatever the arithmetic. Rows at the same distance on
    paper with different differences, such as (1, 3, 3) and (3, 3, 1), can be separated by
    rounding, and the Contract accepts either order, so a query with such a pair is not
    compared.
    """
    seen = {}
    for row in X:
        gaps = tuple(abs(int(a) - int(b)) for a, b in zip(row, q, strict=True))
        if seen.setdefault(sum(gap * gap for gap in gaps), gaps) != gaps:
            return False
    return True


def test_matches_brute_force_reference_on_random_small_integer_data():
    # A grid of real values: the band, from -3 to 3, that a real measurement falls in. Whole
    # numbers on a small grid give many equal distances and many vote ties.
    compared = 0
    for seed in range(120):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 25))
        d = int(rng.integers(1, 4))
        X = _real_ints(rng, -3, 4, (n, d))
        y = _real_ints(rng, 0, 4, n)
        Q = _real_ints(rng, -3, 4, (10, d))
        k = int(rng.integers(1, n + 1))
        keep = [index for index, q in enumerate(Q.tolist()) if _ties_are_exact(X.tolist(), q)]
        compared += len(keep)
        np.testing.assert_array_equal(
            _call(X, y, Q, k)[keep],
            _reference(X.tolist(), y.tolist(), Q[keep].tolist(), k),
            err_msg="predictions differ from the brute-force vote: check the neighbour order "
            "(distance, then index) and the k neighbours used in the vote",
        )
    assert compared >= 300, "the grid cases left too few queries to compare: check the fixture"


def test_matches_brute_force_reference_on_random_float_data():
    for seed in range(20):
        rng = data_rng(f"knn-classify-p01:float:{seed}")
        n = int(rng.integers(5, 40))
        points, value = _houses(rng, n + 15, 3)
        X, Q = points[:n], points[n:]
        y = _value_band(value[:n], lowest=-2)
        k = int(rng.integers(1, n + 1))
        np.testing.assert_array_equal(
            _call(X, y, Q, k),
            _reference(X.tolist(), y.tolist(), Q.tolist(), k),
            err_msg="predictions differ from the brute-force vote on continuous features",
        )


def test_matches_scikit_learn_brute_force_on_tie_free_binary_data():
    points, value = _houses(data_rng("knn-classify-p01:scikit-learn"), 180, 3)
    X, Q = points[:120], points[120:]
    y = (value[:120] > 1.8).astype(int)  # above or below 1.8, near the median value
    for k in (1, 3, 5, 11):
        ref = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X, y).predict(Q)
        np.testing.assert_array_equal(
            _call(X, y, Q, k),
            ref,
            err_msg="predictions differ from scikit-learn brute-force KNN on binary data, odd k",
        )


def test_equal_distances_on_a_grid_are_ordered_by_training_index():
    xs, ys = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    X = np.column_stack([xs.ravel(), ys.ravel()])
    rng = np.random.default_rng(21)
    y = _real_ints(rng, 0, 3, len(X))
    Q = np.array([[0, 0], [1, 1], [-2, 1], [3, -3]])
    for k in (1, 2, 4, 7, 12):
        np.testing.assert_array_equal(
            _call(X, y, Q, k),
            _reference(X.tolist(), y.tolist(), Q.tolist(), k),
            err_msg="equal distances must be broken by the lower training index",
        )


def test_vote_ties_go_to_the_nearest_tied_label_not_the_smallest():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([9, 2, 9, 2, 1, 1])
    Q = np.array([[1.1], [4.4], [2.9]])
    for k in (2, 4, 6):
        np.testing.assert_array_equal(
            _call(X, y, Q, k),
            _reference(X.tolist(), y.tolist(), Q.tolist(), k),
            err_msg="a vote tie must go to the class of the nearest tied neighbour",
        )


def test_k_equal_to_training_size_votes_over_every_row():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [4.0, 4.0], [3.0, 1.0]])
    y = np.array([3, 3, 8, 8, 8])
    Q = np.array([[0.5, 0.5], [3.5, 3.5]])
    np.testing.assert_array_equal(
        _call(X, y, Q, 5),
        _reference(X.tolist(), y.tolist(), Q.tolist(), 5),
        err_msg="with k equal to the training size every row must enter the vote",
    )


INVALID_K = [
    ("k-zero", 0),
    ("k-above-n", 10),
    ("k-negative", -1),
    ("k-bool", True),
    ("k-float", 2.0),
]


@pytest.mark.parametrize(("k",), [(case[1],) for case in INVALID_K], ids=[c[0] for c in INVALID_K])
def test_invalid_k_raises_value_error(k):
    _expect_value_error(
        "k must be an integer with 1 <= k <= n",
        [[0.0], [1.0], [2.0]],
        [0, 1, 1],
        [[0.4]],
        k,
    )


INVALID_INPUTS = [
    ("one-dimensional-train", np.array([0.0, 1.0]), [0, 1], [[0.2]], 1),
    ("column-mismatch", [[0.0, 1.0], [1.0, 0.0]], [0, 1], [[0.2]], 1),
    ("label-length-mismatch", [[0.0], [1.0]], [0, 1, 1], [[0.2]], 1),
    ("float-labels", [[0.0], [1.0]], [0.0, 1.0], [[0.2]], 1),
    ("nan-feature", [[0.0], [np.nan]], [0, 1], [[0.2]], 1),
    ("infinite-query", [[0.0], [1.0]], [0, 1], [[np.inf]], 1),
]


@pytest.mark.parametrize(
    ("X", "y", "Q", "k"),
    [case[1:] for case in INVALID_INPUTS],
    ids=[case[0] for case in INVALID_INPUTS],
)
def test_invalid_shapes_dtypes_and_values_raise_value_error(X, y, Q, k):
    _expect_value_error(
        "inputs must be 2-D features, integer labels, and finite values", X, y, Q, k
    )


def test_empty_query_returns_empty_integer_array():
    X = np.array([[0.0], [1.0]])
    out = knn_classify(X, np.array([0, 1]), np.zeros((0, 1)), k=1)
    assert np.asarray(out).shape == (0,), "an empty query must return an empty 1-D array"


def test_translating_every_point_does_not_change_any_prediction():
    points, value = _houses(data_rng("knn-classify-p01:translation"), 100, 3)
    X, Q = points[:60], points[60:]
    y = _value_band(value[:60])
    base = _call(X, y, Q, 5)
    shifted = _call(X + 1e8, y, Q + 1e8, 5)
    np.testing.assert_array_equal(
        shifted,
        base,
        err_msg="predictions change when all points are translated: compute x - q directly, "
        "not with the expansion |x|^2 - 2 x.q + |q|^2, which loses precision at large offsets",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_prediction_is_always_a_training_label(seed):
    rng = data_rng(f"knn-classify-p01:training-label:{seed}")
    n = int(rng.integers(1, 12))
    points, value = _houses(rng, n + 5, 2)
    X, Q = points[:n], points[n:]
    y = _value_band(value[:n], lowest=-5) * 3
    k = int(rng.integers(1, n + 1))
    out = _call(X, y, Q, k)
    assert set(out.tolist()) <= set(y.tolist()), "a prediction must be one of the training labels"


def test_inputs_are_not_mutated():
    points, value = _houses(data_rng("knn-classify-p01:not-mutated"), 25, 2)
    X, Q = points[:20], points[20:]
    y = _value_band(value[:20])
    X0, y0, Q0 = X.copy(), y.copy(), Q.copy()
    _call(X, y, Q, 3)
    np.testing.assert_array_equal(X, X0, err_msg="X_train was modified in place")
    np.testing.assert_array_equal(y, y0, err_msg="y_train was modified in place")
    np.testing.assert_array_equal(Q, Q0, err_msg="X_query was modified in place")


def test_large_finite_distances_keep_the_nearest_neighbour_order():
    got = _call([[2e200], [1e200]], [9, 3], [[0.0]], k=1)
    assert np.array_equal(got, [3]), (
        "distance overflow must not turn unequal finite distances into an index tie"
    )


def test_zero_neighbours_are_rejected_even_when_there_are_no_queries():
    try:
        knn_classify([[0.0], [1.0]], [0, 1], np.zeros((0, 1)), 0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"k of zero must raise ValueError; the call raised {type(exc).__name__}")
    pytest.fail(
        "k of zero was accepted: k must be between 1 and the number of training rows, even when"
        " the query is empty"
    )


def test_training_rows_without_feature_columns_are_rejected():
    try:
        knn_classify(np.zeros((3, 0)), [0, 1, 0], np.zeros((2, 0)), 1)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"training features with no columns must raise ValueError; the call raised"
            f" {type(exc).__name__}"
        )
    pytest.fail(
        "training features with no columns were accepted: X_train needs at least one feature column"
    )


def test_returned_array_holds_integer_labels():
    # The Contract promises a 1-D integer array of training labels, not floats.
    out = knn_classify([[0.0], [1.0]], [4, 2], [[0.2]], k=1)
    assert np.issubdtype(np.asarray(out).dtype, np.integer), (
        "the returned array must hold integer class labels, not floats"
    )


def test_predicted_labels_come_back_as_a_numpy_array():
    X = [[0, 0], [1, 0], [0, 1], [5, 5]]
    out = knn_classify(X, [0, 1, 1, 0], [[0.2, 0.2]], k=3)
    assert isinstance(out, np.ndarray), (
        "the result must be a NumPy array: a list of the same numbers is another type"
    )
