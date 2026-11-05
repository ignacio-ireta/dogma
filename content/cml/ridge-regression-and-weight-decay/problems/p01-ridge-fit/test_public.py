import numpy as np
import pytest
from submission import ridge_fit


def test_worked_example_with_penalty_two_shrinks_the_slope_to_a_quarter():
    intercept, coef = ridge_fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 2.0], 2.0)
    np.testing.assert_allclose(
        coef,
        [0.25],
        atol=1e-12,
        err_msg="slope must be sum(x_c * y_c) / (sum(x_c^2) + alpha) = 1 / (2 + 2)",
    )
    np.testing.assert_allclose(
        intercept,
        5.0 / 3.0 - 2.0 * 0.25,
        atol=1e-12,
        err_msg="intercept must be mean(y) - mean(X) @ coef, so the line passes through the means",
    )


def test_rank_deficient_design_is_fine_when_alpha_is_positive():
    intercept, coef = ridge_fit([[1.0, 3.0], [2.0, 6.0], [3.0, 9.0]], [1.0, 2.0, 3.0], 1.0)
    np.testing.assert_allclose(
        coef,
        [2.0 / 21.0, 6.0 / 21.0],
        atol=1e-12,
        err_msg="with alpha > 0 the centred system is solvable even when the columns are dependent",
    )
    np.testing.assert_allclose(
        intercept,
        2.0 / 21.0,
        atol=1e-12,
        err_msg="intercept must be mean(y) - mean(X) @ coef",
    )


def test_negative_alpha_raises_value_error():
    try:
        ridge_fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 2.0], -0.5)
    except ValueError:
        return
    pytest.fail("a negative alpha was accepted: the penalty must be >= 0")
