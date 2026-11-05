import numpy as np
import pytest
from submission import fit_linear_regression


def test_exact_line_recovers_intercept_and_slope():
    intercept, coef = fit_linear_regression([[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0])
    np.testing.assert_allclose(
        [intercept, *coef],
        [1.0, 2.0],
        atol=1e-9,
        err_msg="y = 1 + 2x should give intercept 1 and slope 2: check the intercept column",
    )


def test_no_intercept_fits_a_line_through_the_origin():
    intercept, coef = fit_linear_regression(
        [[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], fit_intercept=False
    )
    assert intercept == 0.0, "without an intercept the returned intercept must be 0.0"
    np.testing.assert_allclose(coef, [2.0], atol=1e-9)


def test_scaled_duplicate_column_is_rank_deficient():
    try:
        fit_linear_regression([[1.0, 3.0], [2.0, 6.0], [3.0, 9.0]], [1.0, 2.0, 3.0])
    except ValueError:
        return
    pytest.fail("a scaled duplicate column (rank deficient) was accepted: check the rank first")


def test_one_dimensional_x_raises_value_error():
    try:
        fit_linear_regression([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    except ValueError:
        return
    pytest.fail("a 1-D X must raise ValueError: X has to be 2-D, one row per sample")
