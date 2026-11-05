import numpy as np
import pytest
from submission import lasso_cd


def test_worked_example_soft_threshold_divided_by_column_curvature():
    coef = lasso_cd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 1.0)
    np.testing.assert_allclose(
        coef,
        [25.0 / 14.0],
        atol=1e-9,
        err_msg="one coordinate: w = S(rho, alpha) / z with rho = 28/3, z = 14/3, so (28/3 - 1) / "
        "(14/3)",
    )


def test_penalty_above_the_correlation_gives_exact_zero():
    coef = lasso_cd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 10.0)
    assert coef.shape == (1,), "the coefficient vector must have one entry per column"
    assert coef[0] == 0.0, (
        "a correlation smaller than alpha in size must give an exact zero: "
        "soft-thresholding sets it to 0, it does not shrink it to a small number"
    )


def test_negative_alpha_raises_value_error():
    try:
        lasso_cd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], -1.0)
    except ValueError:
        return
    pytest.fail("a negative alpha was accepted: the penalty must be >= 0")
