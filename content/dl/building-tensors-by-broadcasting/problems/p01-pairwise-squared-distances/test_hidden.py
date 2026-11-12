import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import cdist
from submission import pairwise_sq_distances

from dogma_data import rng as data_rng
from dogma_data import sample_rows

FINITE = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


def _real_rows(gen, n, d):
    """n real rows with d standardised features: breast_cancer, or digits when d is above 30.

    The features are correlated and skewed, and many values repeat, as measured data does.
    """
    name = "breast_cancer" if d <= 30 else "digits"
    return sample_rows(name, gen, n, d, standardize=True)[0]


@st.composite
def point_clouds(draw, max_n=6, max_d=4):
    """One point cloud X of shape (n, d), with n >= 1 and d >= 1."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    d = draw(st.integers(min_value=1, max_value=max_d))
    return draw(arrays(np.float64, (n, d), elements=FINITE))


@st.composite
def cloud_pairs(draw, max_n=6, max_d=4):
    """Two clouds X (n, d) and Y (m, d) that share the feature count d."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    m = draw(st.integers(min_value=1, max_value=max_n))
    d = draw(st.integers(min_value=1, max_value=max_d))
    X = draw(arrays(np.float64, (n, d), elements=FINITE))
    Y = draw(arrays(np.float64, (m, d), elements=FINITE))
    return X, Y


def _assert_close(actual, desired, rtol=1e-7, atol=0.0, err_msg=""):
    """Like np.testing.assert_allclose, but the message is the first line of the failure."""
    got = np.asarray(actual, dtype=np.float64)
    want = np.asarray(desired, dtype=np.float64)
    if want.ndim == 0:
        want = np.broadcast_to(want, got.shape)
    if got.shape != want.shape:
        raise AssertionError(f"{err_msg}: shape {got.shape}, expected {want.shape}")
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        worst = float(np.max(np.abs(got - want)))
        raise AssertionError(f"{err_msg} (largest difference {worst:.3g})")


def _assert_equal(actual, desired, err_msg=""):
    """Exact array equality, with the message first, as in _assert_close."""
    if not np.array_equal(np.asarray(actual), np.asarray(desired)):
        raise AssertionError(err_msg)


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_matches_scipy_cdist_on_random_gaussian_points():
    rng = data_rng("pairwise-sq-distances-p01:cdist")
    shapes = [(5, 5, 3), (4, 7, 3), (7, 4, 3), (1, 6, 2), (6, 1, 2), (3, 4, 1), (10, 12, 50)]
    for n, m, d in shapes:
        points = _real_rows(rng, n + m, d)
        X, Y = points[:n], points[n:]
        _assert_close(
            pairwise_sq_distances(X, Y),
            cdist(X, Y, "sqeuclidean"),
            rtol=1e-9,
            atol=1e-9,
            err_msg=(
                "differs from cdist: check the (n, 1, d) and (1, m, d) broadcast axes "
                "and the sum over features"
            ),
        )


def test_points_far_from_origin_keep_their_small_distances():
    points = _real_rows(data_rng("pairwise-sq-distances-p01:offset"), 11, 3)
    X = 1e8 + points[:6]
    Y = 1e8 + points[6:]
    _assert_close(
        pairwise_sq_distances(X, Y),
        cdist(X, Y, "sqeuclidean"),
        rtol=1e-6,
        atol=1e-9,
        err_msg=(
            "lost precision far from the origin: ||x||^2 - 2x.y + ||y||^2 cancels large terms; "
            "subtract the points directly, or centre both sets on a shared point first"
        ),
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(point_clouds())
def test_self_distance_matrix_is_symmetric_with_zero_diagonal(X):
    D = pairwise_sq_distances(X, X)
    scale = 1.0 + float(np.sum(X * X))
    _assert_close(D, D.T, rtol=1e-9, atol=1e-9 * scale, err_msg="D(X, X) should be symmetric")
    _assert_close(
        np.diag(D),
        0.0,
        atol=1e-9 * scale,
        err_msg="a point is at distance 0 from itself: the diagonal of D(X, X) should be 0",
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(cloud_pairs())
def test_every_entry_is_non_negative(pair):
    X, Y = pair
    D = pairwise_sq_distances(X, Y)
    assert D.min() >= 0.0, (
        f"negative squared distance {D.min()}: rounding in the expansion needs clipping at zero"
    )


def test_self_distances_stay_non_negative_for_large_coordinates():
    # Identical points must come out exactly non-negative. Rounding in the expansion can turn a
    # true zero into a small negative number, and a square root of that is NaN.
    X = _real_rows(data_rng("pairwise-sq-distances-p01:non-negative"), 200, 3) * 1e3
    D = pairwise_sq_distances(X, X)
    assert D.min() >= 0.0, (
        "a squared distance came out negative: rounding in an expansion needs clipping at zero"
    )


def test_no_python_loop_over_rows_or_pairs():
    # Broadcasting runs a fixed number of lines however many points there are. A loop over rows
    # or pairs runs at least one line per row or pair, so 200 points would run hundreds of them.
    points = _real_rows(data_rng("pairwise-sq-distances-p01:no-loop"), 400, 3)
    X, Y = points[:200], points[200:]
    executed = 0

    def count_lines(frame, event, arg):
        nonlocal executed
        if not frame.f_code.co_filename.endswith("submission.py"):
            return None
        if event == "line":
            executed += 1
        return count_lines

    sys.settrace(count_lines)
    try:
        pairwise_sq_distances(X, Y)
    finally:
        sys.settrace(None)
    assert executed < 100, (
        "a Python loop over rows or pairs: build the result with broadcasting, not per-pair code"
    )


def test_empty_point_sets_give_matrices_with_the_right_shape():
    assert pairwise_sq_distances(np.zeros((0, 3)), np.ones((4, 3))).shape == (0, 4), (
        "shape mismatch: no points in X should give a (0, 4) result"
    )
    assert pairwise_sq_distances(np.ones((2, 3)), np.zeros((0, 3))).shape == (2, 0), (
        "shape mismatch: no points in Y should give a (2, 0) result"
    )


def test_feature_counts_must_match_even_when_one_side_has_a_single_feature():
    _raised_value_error(
        lambda: pairwise_sq_distances(np.zeros((3, 1)), np.zeros((4, 3))),
        "X has 1 feature and Y has 3: broadcasting would silently stretch X, so raise ValueError",
    )


def test_one_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: pairwise_sq_distances(np.zeros(3), np.zeros((2, 3))),
        "X is 1-D (one point), not (n, d); raise ValueError instead of broadcasting it",
    )


def test_integer_input_gives_float64_matrix_of_shape_n_by_m():
    D = pairwise_sq_distances(np.array([[0, 0], [3, 0]]), np.array([[0, 4]]))
    assert isinstance(D, np.ndarray), "return a NumPy array, not a list"
    assert D.dtype == np.float64, (
        f"dtype mismatch: expected float64, got {D.dtype}; integer input must not truncate"
    )
    assert D.shape == (2, 1), (
        "shape mismatch: the result should have one row per point of X and one column per "
        "point of Y"
    )
    _assert_close(D, [[16.0], [25.0]], err_msg="integer input gave wrong distances")


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[0.0, 1.0], [2.0, 2.0], [5.0, 5.0]])
    X_before, Y_before = X.copy(), Y.copy()
    pairwise_sq_distances(X, Y)
    _assert_equal(X, X_before, err_msg="input X was modified in place")
    _assert_equal(Y, Y_before, err_msg="input Y was modified in place")
