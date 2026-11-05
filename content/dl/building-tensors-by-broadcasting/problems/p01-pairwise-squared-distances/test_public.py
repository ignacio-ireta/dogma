import numpy as np
import pytest
from submission import pairwise_sq_distances


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_two_points_against_one_point():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Y = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(
        pairwise_sq_distances(X, Y),
        [[1.0], [2.0]],
        atol=1e-12,
        err_msg="(0, 0) to (0, 1) is 1 and (1, 0) to (0, 1) is 2: check the (n, m) layout",
    )


def test_self_distances_are_zero_on_the_diagonal():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        pairwise_sq_distances(X, X),
        [[0.0, 8.0], [8.0, 0.0]],
        atol=1e-12,
        err_msg="a point is at distance 0 from itself, and (1, 2) to (3, 4) is 4 + 4 = 8",
    )


def test_one_feature_points_give_squared_gaps():
    X = np.array([[0.0], [2.0]])
    Y = np.array([[1.0], [5.0]])
    np.testing.assert_allclose(
        pairwise_sq_distances(X, Y),
        [[1.0, 25.0], [1.0, 9.0]],
        atol=1e-12,
        err_msg="d = 1: each entry is the square of the gap; check the feature-axis sum",
    )


def test_feature_count_mismatch_raises_value_error():
    _raised_value_error(
        lambda: pairwise_sq_distances(np.zeros((2, 3)), np.zeros((4, 2))),
        "X and Y have different feature counts, so pairwise_sq_distances must raise ValueError",
    )
