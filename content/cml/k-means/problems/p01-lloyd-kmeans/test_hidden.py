import itertools
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.cluster import KMeans
from submission import kmeans

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


def _points(gen, n, d):
    """n real block groups of california_housing with d per-household averages, standardised.

    The averages are ratios of counts, so no two distances tie exactly, and they are
    heavy-tailed: a few block groups sit far from everything else, as real outliers do.
    """
    X, _ = sample_rows("california_housing", gen, n, standardize=True)
    return X[:, RATIO_COLUMNS[:d]]


def _call(X, C0, **kwargs):
    """Run with NumPy warnings as errors; a crash on valid input is a contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = kmeans(X, C0, **kwargs)
    except RuntimeWarning:
        pytest.fail("the update emitted a NumPy warning: guard the mean of an empty cluster")
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    if not (isinstance(result, tuple) and len(result) == 4):
        pytest.fail("return a four-item tuple (centroids, labels, inertia, n_iter)")
    if np.shape(result[0]) != np.shape(C0):
        pytest.fail(
            "shape mismatch: centroids must keep the shape of init_centroids, one row per cluster"
        )
    return result


def _expect_value_error(mistake, X, C0, **kwargs):
    try:
        kmeans(X, C0, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a result instead of raising ValueError")


def _reference(X, C0, max_iter, tol):
    """Plain-loop Lloyd's algorithm written independently of the learner's code."""
    n, k = len(X), len(C0)
    C = [list(map(float, row)) for row in C0]
    n_iter = 0

    def nearest(point, cents):
        best, best_d = 0, None
        for j, c in enumerate(cents):
            d = math.fsum((a - b) ** 2 for a, b in zip(point, c, strict=True))
            if best_d is None or d < best_d:
                best, best_d = j, d
        return best

    for iteration in range(1, max_iter + 1):
        labels = [nearest(X[i], C) for i in range(n)]
        new = []
        for j in range(k):
            members = [X[i] for i in range(n) if labels[i] == j]
            if members:
                new.append([math.fsum(col) / len(members) for col in zip(*members, strict=True)])
            else:
                new.append(list(C[j]))
        shift = max(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(nc, oc, strict=True)))
            for nc, oc in zip(new, C, strict=True)
        )
        C = new
        n_iter = iteration
        if shift <= tol:
            break
    labels = [nearest(X[i], C) for i in range(n)]
    inertia = math.fsum(
        sum((a - b) ** 2 for a, b in zip(X[i], C[labels[i]], strict=True)) for i in range(n)
    )
    return np.array(C), np.array(labels), inertia, n_iter


def test_matches_scikit_learn_lloyd_on_well_separated_blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([c + _real_normal(rng, (30, 2), scale=0.5) for c in centres])
    C0 = centres + np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    ref = KMeans(n_clusters=3, init=C0, n_init=1, algorithm="lloyd", tol=1e-14, max_iter=500).fit(X)
    C, labels, inertia, _ = _call(X, C0, max_iter=500, tol=1e-12)
    np.testing.assert_allclose(
        C,
        ref.cluster_centers_,
        atol=1e-8,
        err_msg="centroids differ from scikit-learn's Lloyd run: check the update and means",
    )
    np.testing.assert_array_equal(
        labels, ref.labels_, err_msg="labels differ from scikit-learn's Lloyd run"
    )
    np.testing.assert_allclose(
        inertia,
        ref.inertia_,
        rtol=1e-9,
        err_msg="inertia differs: sum squared distances to the centroids",
    )


def test_matches_independent_reference_including_iteration_count():
    for seed in range(12):
        rng = data_rng(f"lloyd-kmeans-p01:{100 + seed}")
        n = int(rng.integers(6, 40))
        d = int(rng.integers(1, 4))
        k = int(rng.integers(1, min(5, n) + 1))
        X = _points(rng, n, d)
        C0 = X[rng.choice(n, size=k, replace=False)]
        max_iter, tol = 50, 1e-6
        want_C, want_labels, want_inertia, want_iter = _reference(
            X.tolist(), C0.tolist(), max_iter, tol
        )
        C, labels, inertia, n_iter = _call(X, C0, max_iter=max_iter, tol=tol)
        assert n_iter == want_iter, (
            "iteration count differs: stop once the largest centroid shift is at most tol, "
            "and count the iterations run"
        )
        np.testing.assert_array_equal(
            labels, want_labels, err_msg="labels differ from the nearest-centroid rule"
        )
        np.testing.assert_allclose(
            C, want_C, atol=1e-9, err_msg="centroids differ from the Lloyd update"
        )
        np.testing.assert_allclose(
            inertia, want_inertia, rtol=1e-9, err_msg="inertia differs from the reference"
        )


def test_empty_cluster_keeps_its_centroid_on_random_data():
    rng = data_rng(f"lloyd-kmeans-p01:{200}")
    X = _points(rng, 20, 2)
    far = np.array([[1000.0, 1000.0]])
    C0 = np.vstack([X[0], far])
    C, labels, _, _ = _call(X, C0, max_iter=20, tol=1e-9)
    np.testing.assert_array_equal(
        C[1], far[0], err_msg="a centroid with no samples must keep its previous position"
    )
    assert np.all(labels == 0), (
        "an empty cluster must keep its old centroid; returned labels must refe"
        "r to the nearest final centroid"
    )


def test_exact_ties_go_to_the_lower_centroid_index():
    X = [[1.0, 0.0], [5.0, 0.0]]
    C, labels, inertia, n_iter = _call(X, [[0.0, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(
        labels, [0, 1], err_msg="an exact tie must go to the lower centroid index"
    )
    np.testing.assert_allclose(
        C,
        [[1.0, 0.0], [5.0, 0.0]],
        atol=1e-12,
        err_msg="a tie resolved to the wrong centroid changes the fixed point",
    )
    assert inertia == 0.0, "inertia must be zero when every sample sits on its centroid"
    assert n_iter == 2, (
        "a tie resolved to the wrong centroid changes the iteration count of this run"
    )


def test_stops_when_the_largest_shift_equals_tol():
    X = [[0.0, 0.0], [0.0, 1.0]]
    _, _, _, n_iter = _call(X, [[0.0, 0.0]], max_iter=50, tol=0.5)
    assert n_iter == 1, (
        "the stopping test is shift <= tol: a shift of exactly tol must stop the run"
    )


def test_max_iter_caps_the_number_of_iterations():
    rng = data_rng(f"lloyd-kmeans-p01:{300}")
    X = _points(rng, 60, 2)
    C0 = X[:3]
    want = _reference(X.tolist(), C0.tolist(), 3, 0.0)
    _, _, _, n_iter = _call(X, C0, max_iter=3, tol=0.0)
    assert n_iter == want[3] == 3, (
        "max_iter must cap the number of iterations run, even when the shift never meets tol"
    )


def test_returned_labels_match_the_returned_centroids():
    X = [[0.4, 0.0], [0.6, 0.0], [10.0, 0.0]]
    C, labels, inertia, _ = _call(X, [[0.0, 0.0], [1.0, 0.0]], max_iter=1)
    np.testing.assert_allclose(
        C,
        [[0.4, 0.0], [5.3, 0.0]],
        atol=1e-12,
        err_msg="returned centroids must contain the means from the final completed update",
    )
    np.testing.assert_array_equal(
        labels,
        [0, 0, 1],
        err_msg="labels must be the nearest-centroid assignment to the returned centroids, not the "
        "labels from before the last update",
    )
    expected = float(np.sum((np.array(X) - C[labels]) ** 2))
    assert inertia == pytest.approx(expected, rel=1e-12), (
        "inertia must be computed against the returned centroids"
    )


def test_inertia_never_increases_as_more_iterations_are_allowed():
    for seed in range(8):
        rng = data_rng(f"lloyd-kmeans-p01:{400 + seed}")
        X = _points(rng, 50, 2)
        C0 = X[rng.choice(50, size=4, replace=False)]
        values = [_call(X, C0, max_iter=m, tol=0.0)[2] for m in range(1, 9)]
        for before, after in itertools.pairwise(values):
            assert after <= before + 1e-9, (
                "inertia rose between iterations: Lloyd's assign and update steps never increase it"
            )


def test_large_common_offset_does_not_change_the_clustering():
    rng = data_rng(f"lloyd-kmeans-p01:{500}")
    X = _points(rng, 60, 3)
    C0 = X[:4].copy()
    base_C, base_labels, _, base_iter = _call(X, C0, max_iter=100, tol=1e-9)
    off = 1e7
    C, labels, _, n_iter = _call(X + off, C0 + off, max_iter=100, tol=1e-9)
    np.testing.assert_array_equal(
        labels, base_labels, err_msg="a common offset changed the labels: compute x - c directly"
    )
    np.testing.assert_allclose(
        C - off,
        base_C,
        atol=1e-6,
        err_msg="a common offset changed the centroids beyond rounding: avoid the expansion "
        "|x|^2 - 2 x.c + |c|^2",
    )
    assert n_iter == base_iter, "a common offset changed the number of iterations"


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=5000))
def test_centroids_stay_inside_the_bounding_box_of_the_samples(seed):
    rng = data_rng(f"lloyd-kmeans-p01:{seed}")
    n = int(rng.integers(2, 20))
    X = _points(rng, n, 2)
    k = int(rng.integers(1, n + 1))
    C0 = X[:k]
    cents, _, _, _ = _call(X, C0, max_iter=30, tol=1e-9)
    low, high = X.min(axis=0) - 1e-9, X.max(axis=0) + 1e-9
    assert np.all((cents >= low) & (cents <= high)), (
        "a centroid left the bounding box of the samples: an update must only average samples"
    )


def test_stopping_uses_the_largest_single_move_as_a_length():
    for seed in range(8):
        rng = data_rng(f"lloyd-kmeans-p01:{700 + seed}")
        X = _points(rng, 40, 2) * 3
        C0 = X[rng.choice(40, size=4, replace=False)]
        for tol in (0.3, 0.5, 1.0):
            want = _reference(X.tolist(), C0.tolist(), 200, tol)[3]
            got = _call(X, C0, max_iter=200, tol=tol)[3]
            assert got == want, (
                "iteration count differs under a coarse tol: stop when the largest single centroid "
                "move (a Euclidean length) is at most tol, not the total or the squared move"
            )


def test_stopping_compares_the_shift_itself_not_its_square():
    X = [[0.0, 0.0], [0.0, 1.0]]
    _, _, _, n_iter = _call(X, [[0.0, 0.0]], max_iter=50, tol=0.25)
    assert n_iter == 2, "compare the Euclidean centroid shift itself with tol, not its square"


INVALID = [
    ("too-many-centroids", [[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0], [3.0]], {}),
    ("one-dimensional-x", [0.0, 1.0], [[0.0]], {}),
    ("column-mismatch", [[0.0, 1.0], [1.0, 0.0]], [[0.0]], {}),
    ("nan-sample", [[0.0], [float("nan")]], [[0.0]], {}),
    ("infinite-centroid", [[0.0], [1.0]], [[float("inf")]], {}),
    ("zero-max-iter", [[0.0], [1.0]], [[0.0]], {"max_iter": 0}),
    ("negative-tol", [[0.0], [1.0]], [[0.0]], {"tol": -1.0}),
    ("empty-centroids", [[0.0], [1.0]], np.zeros((0, 1)), {}),
]


@pytest.mark.parametrize(
    ("X", "C0", "kwargs"),
    [case[1:] for case in INVALID],
    ids=[case[0] for case in INVALID],
)
def test_invalid_inputs_raise_value_error(X, C0, kwargs):
    _expect_value_error(
        "inputs must be finite 2-D arrays with matching columns, 1 <= k <= n", X, C0, **kwargs
    )


def test_inputs_are_not_mutated():
    rng = data_rng(f"lloyd-kmeans-p01:{600}")
    X = _points(rng, 15, 2)
    C0 = X[:3].copy()
    X0, C00 = X.copy(), C0.copy()
    _call(X, C0, max_iter=20, tol=1e-6)
    np.testing.assert_array_equal(X, X0, err_msg="X was modified in place")
    np.testing.assert_array_equal(C0, C00, err_msg="init_centroids was modified in place")


def test_fewer_columns_in_X_than_in_centroids_raises_value_error():
    # Broadcasting would silently accept this shape pair, so only the column check can raise.
    _expect_value_error(
        "init_centroids must have the same number of columns as X",
        [[0.0], [1.0], [2.0]],
        [[0.0, 0.0]],
    )


def test_zero_column_samples_raise_value_error():
    _expect_value_error(
        "X must have at least one column",
        np.zeros((3, 0)),
        np.zeros((1, 0)),
    )


def test_default_max_iter_and_tol_follow_the_signature():
    import inspect  # noqa: PLC0415

    parameters = inspect.signature(kmeans).parameters
    assert parameters["max_iter"].default == 300, "the default max_iter must be 300"
    assert parameters["tol"].default == 1e-6, "the default tol must be 1e-6"


def test_one_sample_with_one_centroid_is_accepted():
    # One sample and one centroid at the same point: the centroid stays put, and the first
    # iteration already meets the stopping rule, so exactly one iteration runs.
    centroids, labels, inertia, n_iter = _call([[1.0, 2.0]], [[1.0, 2.0]])
    np.testing.assert_allclose(
        centroids,
        [[1.0, 2.0]],
        atol=1e-12,
        err_msg="a single sample is valid input: its centroid must stay on that sample",
    )
    np.testing.assert_array_equal(labels, [0], err_msg="a single sample must get label 0")
    assert inertia == 0.0, "inertia is wrong: a centroid sitting on its only sample costs nothing"
    assert n_iter == 1, "iteration count is wrong: the first update already meets the stopping rule"


def test_boolean_max_iter_is_rejected():
    # Python treats True as the integer 1, but a boolean is not an integer in this contract.
    _expect_value_error(
        "a boolean passed as max_iter was accepted; booleans are not integers here",
        [[0.0], [1.0]],
        [[0.0]],
        max_iter=True,
    )


def test_inertia_is_a_float_and_not_an_array():
    _, _, inertia, _ = kmeans([[0, 0], [0, 1], [10, 10], [10, 11]], [[0, 0], [10, 10]])
    assert isinstance(inertia, float), (
        "inertia must be one float: an array that holds one number is another type"
    )
