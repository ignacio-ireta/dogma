import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.neighbors import KNeighborsClassifier
from submission import knn_classify


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


def test_matches_brute_force_reference_on_random_small_integer_data():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 25))
        d = int(rng.integers(1, 4))
        X = rng.integers(-3, 4, size=(n, d))
        y = rng.integers(0, 4, size=n)
        Q = rng.integers(-3, 4, size=(10, d))
        k = int(rng.integers(1, n + 1))
        np.testing.assert_array_equal(
            _call(X, y, Q, k),
            _reference(X.tolist(), y.tolist(), Q.tolist(), k),
            err_msg="predictions differ from the brute-force vote: check the neighbour order "
            "(distance, then index) and the k neighbours used in the vote",
        )


def test_matches_brute_force_reference_on_random_float_data():
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        n = int(rng.integers(5, 40))
        X = rng.normal(size=(n, 3))
        y = rng.integers(-2, 3, size=n)
        Q = rng.normal(size=(15, 3))
        k = int(rng.integers(1, n + 1))
        np.testing.assert_array_equal(
            _call(X, y, Q, k),
            _reference(X.tolist(), y.tolist(), Q.tolist(), k),
            err_msg="predictions differ from the brute-force vote on continuous features",
        )


def test_matches_scikit_learn_brute_force_on_tie_free_binary_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    Q = rng.normal(size=(60, 4))
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
    y = rng.integers(0, 3, size=len(X))
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
    rng = np.random.default_rng(31)
    X = rng.normal(size=(60, 3))
    y = rng.integers(0, 3, size=60)
    Q = rng.normal(size=(40, 3))
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
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 12))
    X = rng.normal(size=(n, 2))
    y = rng.integers(-5, 6, size=n)
    Q = rng.normal(size=(5, 2))
    k = int(rng.integers(1, n + 1))
    out = _call(X, y, Q, k)
    assert set(out.tolist()) <= set(y.tolist()), "a prediction must be one of the training labels"


def test_inputs_are_not_mutated():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(20, 2))
    y = rng.integers(0, 3, size=20)
    Q = rng.normal(size=(5, 2))
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
