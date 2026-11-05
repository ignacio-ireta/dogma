import numpy as np
from submission import logloss_grad_hess


def test_zero_margin_gives_half_probability_gradient_and_quarter_hessian():
    grad, hess = logloss_grad_hess([1, 0], [0.0, 0.0])
    np.testing.assert_allclose(
        grad,
        [-0.5, 0.5],
        atol=1e-12,
        err_msg="gradient is wrong at zero margin: it must be sigmoid(margin) - y",
    )
    np.testing.assert_allclose(
        hess,
        [0.25, 0.25],
        atol=1e-12,
        err_msg="hessian is wrong at zero margin: it must be sigmoid * (1 - sigmoid)",
    )


def test_saturated_margins_give_exact_gradient_and_zero_hessian():
    grad, hess = logloss_grad_hess([0, 1], [1e4, -1e4])
    np.testing.assert_allclose(
        grad,
        [1.0, -1.0],
        atol=1e-12,
        err_msg="gradient is wrong at saturated margins: a confidently wrong example has |g| = 1",
    )
    assert np.all(np.isfinite(hess)), (
        "hessian is not finite at saturated margins: avoid exp of a large positive number"
    )
    np.testing.assert_allclose(
        hess,
        [0.0, 0.0],
        atol=1e-12,
        err_msg="hessian must vanish at saturated margins, not blow up or become NaN",
    )


def test_labels_other_than_zero_or_one_raise_value_error():
    try:
        logloss_grad_hess([-1, 1], [0.0, 0.0])
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"labels other than 0 or 1 must raise ValueError; got {type(exc).__name__}"
        ) from exc
    raise AssertionError("labels outside {0, 1} must be rejected, not turned into a gradient")
