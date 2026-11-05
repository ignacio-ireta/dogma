import numpy as np
import pytest
from submission import gaussian_mle


def test_mean_and_biased_covariance_of_three_points():
    mean, cov = gaussian_mle([[1, 2], [3, 4], [5, 9]])
    np.testing.assert_allclose(
        mean,
        [3.0, 5.0],
        atol=1e-9,
        err_msg="the MLE mean is the column mean",
    )
    np.testing.assert_allclose(
        cov,
        [[8 / 3, 14 / 3], [14 / 3, 26 / 3]],
        atol=1e-9,
        err_msg="the MLE covariance divides the centred products by n = 3, not by n - 1",
    )


def test_single_observation_has_zero_covariance():
    mean, cov = gaussian_mle([[1.0, 2.0]])
    np.testing.assert_allclose(
        mean, [1.0, 2.0], atol=1e-12, err_msg="the mean of one point is the point"
    )
    np.testing.assert_allclose(
        cov,
        np.zeros((2, 2)),
        atol=1e-12,
        err_msg="one observation has no spread: the maximum-likelihood covariance is zero",
    )


def test_empty_data_raises_value_error():
    try:
        gaussian_mle(np.empty((0, 2)))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"no observations have no estimate; raised {type(exc).__name__}: {exc}")
    pytest.fail("empty data must raise ValueError: there is no mean to estimate")
