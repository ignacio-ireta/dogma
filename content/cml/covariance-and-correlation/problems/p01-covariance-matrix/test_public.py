import numpy as np
import pytest
from submission import covariance_matrix


def test_two_variables_with_default_ddof_use_n_minus_one():
    np.testing.assert_allclose(
        covariance_matrix([[1, 2], [3, 4], [5, 9]]),
        [[4.0, 7.0], [7.0, 13.0]],
        atol=1e-9,
        err_msg="the default ddof=1 divides the centred products by n - 1 = 2",
    )


def test_ddof_zero_divides_by_n():
    np.testing.assert_allclose(
        covariance_matrix([[1, 2], [3, 4], [5, 9]], ddof=0),
        [[8 / 3, 14 / 3], [14 / 3, 26 / 3]],
        atol=1e-9,
        err_msg="ddof=0 divides the same centred products by n, not by n - 1",
    )


def test_one_row_with_ddof_one_raises_value_error():
    try:
        covariance_matrix([[1.0, 2.0]])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"one observation has no sample covariance; raised {type(exc).__name__}: {exc}")
    pytest.fail("one observation with ddof=1 has no positive divisor: raise ValueError")
