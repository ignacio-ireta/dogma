import numpy as np
import pytest
from submission import mse_sigmoid_grad


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_two_sample_worked_example_gives_one_eighth():
    np.testing.assert_allclose(
        mse_sigmoid_grad(np.array([0.0]), np.array([[1.0], [2.0]]), np.array([1.0, 0.0])),
        [0.125],
        rtol=1e-12,
        err_msg="at w = 0 both predictions are 0.5; residuals (-0.5, 0.5), sigma' = 0.25, so "
        "the gradient is (2/2) * (1*(-0.125) + 2*0.125) = 0.125",
    )


def test_predictions_equal_to_targets_give_zero_gradient():
    np.testing.assert_allclose(
        mse_sigmoid_grad(np.zeros(2), np.eye(2), np.full(2, 0.5)),
        [0.0, 0.0],
        atol=1e-12,
        err_msg="when every prediction equals its target the residual is zero, so the "
        "gradient is zero",
    )


def test_column_vector_y_raises_value_error():
    _raised_value_error(
        lambda: mse_sigmoid_grad(np.zeros(1), np.ones((2, 1)), np.ones((2, 1))),
        "y of shape (n, 1) is not a vector of targets: raise ValueError, do not broadcast",
    )
