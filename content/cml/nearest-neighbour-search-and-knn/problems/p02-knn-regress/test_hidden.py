import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.neighbors import KNeighborsRegressor
from submission import knn_regress

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


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


def _houses(gen, n, d):
    """n real california_housing block groups: d standardised averages and the house value.

    The features are ratios of counts, so no two distances tie exactly, and they are
    heavy-tailed. The value is the median house value in units of 100,000 dollars.
    """
    X, value = sample_rows("california_housing", gen, n, standardize=True)
    return X[:, RATIO_COLUMNS[:d]], value


def _call(X, y, Q, k, **kwargs):
    """Run with NumPy warnings as errors; a crash on valid input is a contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = knn_regress(X, y, Q, k, **kwargs)
    except RuntimeWarning:
        pytest.fail("the distance or average emitted a NumPy warning: guard the zero distance")
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    out = np.asarray(out)
    if out.shape != (len(Q),):
        pytest.fail("shape mismatch: return one prediction per query row, as a 1-D array")
    return out


def _expect_value_error(mistake, X, y, Q, k, **kwargs):
    try:
        knn_regress(X, y, Q, k, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a result instead of raising ValueError")


def _reference(X, y, Q, k, weights):
    """Plain-Python brute force: sort by (distance, index), then average as the contract says."""
    out = []
    for q in Q:
        dist = [math.sqrt(sum((a - b) ** 2 for a, b in zip(x, q, strict=True))) for x in X]
        order = sorted(range(len(X)), key=lambda i: (dist[i], i))[:k]
        d = [dist[i] for i in order]
        yy = [float(y[i]) for i in order]
        if weights == "uniform":
            out.append(sum(yy) / len(yy))
            continue
        exact = [v for dd, v in zip(d, yy, strict=True) if dd == 0.0]
        if exact:
            out.append(sum(exact) / len(exact))
        else:
            w = [1.0 / dd for dd in d]
            out.append(sum(wi * v for wi, v in zip(w, yy, strict=True)) / sum(w))
    return np.array(out, dtype=np.float64)


def test_matches_scikit_learn_for_both_weightings_on_continuous_data():
    points, value = _houses(data_rng("knn-regress-p02:scikit-learn"), 200, 3)
    X, Q, y = points[:150], points[150:], value[:150]
    for weights in ("uniform", "distance"):
        for k in (1, 4, 9):
            ref = (
                KNeighborsRegressor(n_neighbors=k, weights=weights, algorithm="brute")
                .fit(X, y)
                .predict(Q)
            )
            np.testing.assert_allclose(
                _call(X, y, Q, k, weights=weights),
                ref,
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"predictions differ from scikit-learn for weights={weights!r}: check the "
                "neighbour set and the weight 1/d",
            )


def test_matches_brute_force_reference_with_duplicates_and_ties():
    rng = np.random.default_rng(5)
    X = _real_ints(rng, -2, 3, (25, 2)).astype(float)
    y = _real_ints(rng, -4, 5, 25).astype(float)
    Q = _real_ints(rng, -2, 3, (20, 2)).astype(float)
    for weights in ("uniform", "distance"):
        for k in (1, 2, 3, 6, 25):
            np.testing.assert_allclose(
                _call(X, y, Q, k, weights=weights),
                _reference(X.tolist(), y, Q.tolist(), k, weights),
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"predictions differ from the reference for weights={weights!r} "
                "with duplicate rows and equal distances",
            )


def test_zero_distance_rule_ignores_the_other_neighbours_under_distance_weights():
    X = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    y = [10.0, 20.0, -100.0, 300.0]
    out = _call(X, y, [[0.0, 0.0]], k=4, weights="distance")
    np.testing.assert_allclose(
        out,
        [15.0],
        atol=1e-12,
        err_msg="exact matches present: distance weights must average only exact matches",
    )


def test_zero_distance_rule_does_not_apply_under_uniform_weights():
    X = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    y = [10.0, 20.0, -100.0, 300.0]
    out = _call(X, y, [[0.0, 0.0]], k=4, weights="uniform")
    np.testing.assert_allclose(
        out,
        [57.5],
        atol=1e-12,
        err_msg="uniform weights must average all k neighbours, with no zero-distance special case",
    )


def test_k_equal_to_training_size_averages_every_target():
    X = np.array([[0.0], [1.0], [4.0]])
    y = np.array([3.0, 6.0, 9.0])
    np.testing.assert_allclose(
        _call(X, y, [[2.0]], k=3, weights="uniform"),
        [6.0],
        atol=1e-12,
        err_msg="with k equal to the training size the uniform prediction is the global mean",
    )


INVALID_K = [("k-zero", 0), ("k-above-n", 4), ("k-bool", True), ("k-float", 1.5)]


@pytest.mark.parametrize(("k",), [(case[1],) for case in INVALID_K], ids=[c[0] for c in INVALID_K])
def test_invalid_k_raises_value_error(k):
    _expect_value_error(
        "k must be an integer with 1 <= k <= n",
        [[0.0], [1.0], [2.0]],
        [0.0, 1.0, 2.0],
        [[0.4]],
        k,
    )


INVALID_INPUTS = [
    ("one-dimensional-train", np.array([0.0, 1.0]), [0.0, 1.0], [[0.2]]),
    ("column-mismatch", [[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0], [[0.2]]),
    ("target-length-mismatch", [[0.0], [1.0]], [0.0, 1.0, 2.0], [[0.2]]),
    ("nan-target", [[0.0], [1.0]], [0.0, np.nan], [[0.2]]),
    ("nan-feature", [[0.0], [np.nan]], [0.0, 1.0], [[0.2]]),
    ("infinite-query", [[0.0], [1.0]], [0.0, 1.0], [[np.inf]]),
]


@pytest.mark.parametrize(
    ("X", "y", "Q"),
    [case[1:] for case in INVALID_INPUTS],
    ids=[case[0] for case in INVALID_INPUTS],
)
def test_invalid_shapes_and_values_raise_value_error(X, y, Q):
    _expect_value_error(
        "inputs must be finite 2-D features and a 1-D target of matching length", X, y, Q, 1
    )


def test_empty_query_returns_empty_float_array():
    out = knn_regress([[0.0], [1.0]], [0.0, 1.0], np.zeros((0, 1)), k=1, weights="distance")
    assert np.asarray(out).shape == (0,), "an empty query must return an empty 1-D array"


def test_translating_every_point_does_not_change_any_prediction():
    points, value = _houses(data_rng("knn-regress-p02:translation"), 100, 3)
    X, Q, y = points[:60], points[60:], value[:60]
    for weights in ("uniform", "distance"):
        base = _call(X, y, Q, 5, weights=weights)
        shifted = _call(X + 1e8, y, Q + 1e8, 5, weights=weights)
        # A shift of 1e8 leaves about 1e-8 of absolute rounding in each distance, so the
        # tolerance allows that noise. The expansion |x|^2 - 2 x.q + |q|^2 is off by far more.
        np.testing.assert_allclose(
            shifted,
            base,
            rtol=1e-6,
            atol=1e-6,
            err_msg="predictions change when all points are translated: compute x - q directly, "
            "not with the expansion |x|^2 - 2 x.q + |q|^2",
        )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000), st.sampled_from(["uniform", "distance"]))
def test_prediction_lies_between_the_smallest_and_largest_target(seed, weights):
    rng = data_rng(f"knn-regress-p02:between:{seed}")
    n = int(rng.integers(1, 15))
    points, value = _houses(rng, n + 4, 2)
    X, Q, y = points[:n], points[n:], value[:n] * 10.0
    k = int(rng.integers(1, n + 1))
    out = _call(X, y, Q, k, weights=weights)
    assert np.all(out >= y.min() - 1e-9) and np.all(out <= y.max() + 1e-9), (
        "a prediction must be a weighted average of targets, so it lies between their extremes"
    )


def test_inputs_are_not_mutated():
    points, value = _houses(data_rng("knn-regress-p02:not-mutated"), 25, 2)
    X, Q, y = points[:20], points[20:], value[:20]
    X0, y0, Q0 = X.copy(), y.copy(), Q.copy()
    _call(X, y, Q, 3, weights="distance")
    np.testing.assert_array_equal(X, X0, err_msg="X_train was modified in place")
    np.testing.assert_array_equal(y, y0, err_msg="y_train was modified in place")
    np.testing.assert_array_equal(Q, Q0, err_msg="X_query was modified in place")


def test_large_finite_distances_keep_the_weighted_neighbours():
    got = _call([[2e200], [1e200]], [8.0, 2.0], [[0.0]], k=2, weights="distance")
    assert np.allclose(got, [4.0], rtol=1e-12, atol=1e-12), (
        "distance overflow must not destroy inverse-distance weights on finite neighbours"
    )


def test_training_matrix_without_feature_columns_raises_value_error():
    _expect_value_error(
        "X_train must have at least one feature column (d >= 1)",
        [[], []],
        [0.0, 1.0],
        [[]],
        1,
    )


def test_infinite_target_raises_value_error():
    # The Contract wants finite targets; an infinite target must be refused like a NaN one.
    _expect_value_error(
        "an infinite target was accepted", [[0.0], [1.0]], [0.0, np.inf], [[0.2]], 1
    )


def test_read_only_input_arrays_are_accepted_and_left_alone():
    X = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 2.0], [3.0, 1.0]])
    y = np.array([1.5, 0.25, 3.0, 2.0])
    Q = np.array([[0.9, 0.4], [2.5, 1.5]])
    for array in (X, y, Q):
        array.setflags(write=False)
    try:
        knn_regress(X, y, Q, 2, weights="distance")
    except Exception as exc:
        if "read-only" in str(exc):
            pytest.fail(
                "an input array was written to in place: inputs must not be mutated, so work "
                "on a copy",
                pytrace=False,
            )
        pytest.fail(
            "a valid call failed when its input arrays were read-only: a correct solution "
            "only reads its inputs",
            pytrace=False,
        )


def test_predictions_come_back_as_a_numpy_array():
    out = knn_regress([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0], [[1.2]], k=2)
    assert isinstance(out, np.ndarray), (
        "the result must be a NumPy array: a list of the same numbers is another type"
    )
