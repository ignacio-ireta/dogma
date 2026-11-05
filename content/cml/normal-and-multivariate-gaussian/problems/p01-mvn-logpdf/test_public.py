import numpy as np
import pytest
from submission import mvn_logpdf


def test_standard_normal_at_zero_is_minus_half_log_two_pi():
    np.testing.assert_allclose(
        mvn_logpdf([[0.0]], [0.0], [[1.0]]),
        [-0.5 * np.log(2 * np.pi)],
        atol=1e-12,
        err_msg="the standard normal at 0 has log-density -1/2 log(2 pi)",
    )


def test_two_dimensional_identity_at_origin_is_minus_log_two_pi():
    np.testing.assert_allclose(
        mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
        [-np.log(2 * np.pi)],
        atol=1e-12,
        err_msg="two independent standard normals at 0: the constant is -d/2 log(2 pi) with d = 2",
    )


def test_asymmetric_covariance_raises_value_error():
    try:
        mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"an asymmetric covariance is not a covariance; raised {type(exc).__name__}: {exc}"
        )
    pytest.fail("an asymmetric covariance must raise ValueError: check symmetry before factoring")


def test_singular_covariance_raises_value_error():
    try:
        mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a singular covariance has no density; raised {type(exc).__name__}: {exc}")
    pytest.fail("a singular covariance must raise ValueError: it is not positive definite")
