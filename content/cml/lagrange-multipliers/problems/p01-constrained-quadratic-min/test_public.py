import numpy as np
import pytest
from submission import constrained_quadratic_min


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_equal_split_of_a_sum_constraint_with_unit_multiplier():
    x, lam = constrained_quadratic_min(2 * np.eye(2), [0.0, 0.0], [[1.0, 1.0]], [1.0])
    np.testing.assert_allclose(
        x,
        [0.5, 0.5],
        atol=1e-9,
        err_msg="minimising x'x subject to x1 + x2 = 1 should split the constraint equally",
    )
    np.testing.assert_allclose(
        lam,
        [1.0],
        atol=1e-9,
        err_msg="multiplier should equal the slope of the optimal value in b, which is 1 here",
    )


def test_linear_term_moves_the_minimiser_and_sets_the_multiplier_sign():
    x, lam = constrained_quadratic_min(np.eye(2), [-3.0, 0.0], [[1.0, 1.0]], [1.0])
    np.testing.assert_allclose(
        x,
        [2.0, -1.0],
        atol=1e-9,
        err_msg="the linear term c must enter the stationarity condition Qx + c = A'lam",
    )
    np.testing.assert_allclose(
        lam,
        [-1.0],
        atol=1e-9,
        err_msg="multiplier sign mismatch: the Lagrangian is f - lam'(Ax - b)",
    )


def test_rank_deficient_constraints_raise_value_error():
    _raised_value_error(
        lambda: constrained_quadratic_min(
            np.eye(3),
            [0.0, 0.0, 0.0],
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            [1.0, 2.0],
        ),
        "A has dependent rows: the constraints need full row rank, so raise ValueError",
    )
