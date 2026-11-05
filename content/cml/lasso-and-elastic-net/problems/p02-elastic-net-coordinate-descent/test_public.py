import numpy as np
import pytest
from submission import elastic_net_cd


def test_worked_example_mixes_the_l1_threshold_and_the_l2_curvature():
    coef = elastic_net_cd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 1.0, 0.5)
    np.testing.assert_allclose(
        coef,
        [53.0 / 31.0],
        atol=1e-9,
        err_msg="one coordinate: w = S(rho, alpha * l1_ratio) / (z + alpha * (1 - l1_ratio)), "
        "which is (28/3 - 0.5) / (14/3 + 0.5)",
    )


def test_l1_ratio_one_reduces_to_the_lasso_solution():
    coef = elastic_net_cd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 1.0, 1.0)
    np.testing.assert_allclose(
        coef,
        [25.0 / 14.0],
        atol=1e-9,
        err_msg="with l1_ratio = 1 there is no L2 term: the answer must be the lasso, (28/3 - 1) / "
        "(14/3)",
    )


def test_l1_ratio_above_one_raises_value_error():
    try:
        elastic_net_cd([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 1.0, 1.5)
    except ValueError:
        return
    pytest.fail("an l1_ratio above 1 was accepted: the mixing ratio must lie in [0, 1]")
