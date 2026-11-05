import numpy as np
import pytest
from submission import huber_loss_and_grad


def test_mixed_residuals_give_hand_computed_loss_and_gradient():
    loss, grad = huber_loss_and_grad([0.5, -0.5, 3.0], 1.0)
    np.testing.assert_allclose(
        loss,
        (0.125 + 0.125 + 2.5) / 3,
        rtol=1e-12,
        err_msg="the mean loss is (1/n) times the sum of the per-residual Huber losses",
    )
    np.testing.assert_allclose(
        grad,
        np.array([0.5, -0.5, 1.0]) / 3,
        rtol=1e-12,
        err_msg="gradient_i must be psi(r_i) / n: r inside the threshold, delta * sign(r) outside",
    )


def test_single_outlier_is_linear_with_capped_gradient():
    loss, grad = huber_loss_and_grad([-4.0], 2.0)
    np.testing.assert_allclose(
        loss,
        6.0,
        rtol=1e-12,
        err_msg="outside the threshold the loss is delta * (|r| - delta/2), not r^2/2",
    )
    np.testing.assert_allclose(
        grad,
        [-2.0],
        rtol=1e-12,
        err_msg="outside the threshold the gradient is capped at delta * sign(r)",
    )


def test_non_positive_delta_raises_value_error():
    try:
        huber_loss_and_grad([1.0], 0.0)
    except ValueError:
        return
    pytest.fail("delta = 0 was accepted: delta must be a finite number greater than 0")
