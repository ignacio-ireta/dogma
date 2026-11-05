import numpy as np
import pytest
from submission import fit_logistic_regression


def _call(X, y, mistake, **kwargs):
    """Call the fit; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return fit_logistic_regression(X, y, **kwargs)
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


WORKED_X = np.array([[0.0], [1.0], [2.0], [3.0]])
WORKED_Y = np.array([0, 0, 1, 1])


def test_worked_example_matches_the_penalised_optimum():
    intercept, coef = _call(WORKED_X, WORKED_Y, "worked example failed", l2=1.0)
    np.testing.assert_allclose(
        [intercept, coef[0]],
        [-0.5747523, 0.3831682],
        atol=1e-6,
        err_msg="wrong optimum for the worked example: minimise the mean log-loss plus "
        "(l2 / 2) * ||w||^2, with the intercept unpenalised",
    )
    assert coef.shape == (1,), "shape mismatch: coef must have one entry per feature"


def test_gradient_vanishes_at_the_worked_example():
    intercept, coef = _call(WORKED_X, WORKED_Y, "worked example failed", l2=1.0)
    eta = intercept + WORKED_X @ coef
    p = 1.0 / (1.0 + np.exp(-eta))
    grad_b = np.mean(p - WORKED_Y)
    grad_w = WORKED_X.T @ (p - WORKED_Y) / len(WORKED_Y) + 1.0 * coef
    assert abs(grad_b) <= 1e-8 and np.all(np.abs(grad_w) <= 1e-8), (
        "the returned point is not stationary: the gradient of the penalised loss is not zero"
    )


def test_single_class_labels_raise_value_error():
    try:
        fit_logistic_regression(np.array([[0.0], [1.0]]), np.array([1, 1]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"one class must raise ValueError; the function raised {type(exc).__name__}")
    pytest.fail("labels with a single class must raise ValueError, not return a fit")


def test_negative_l2_raises_value_error():
    try:
        fit_logistic_regression(WORKED_X, WORKED_Y, l2=-0.5)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"negative l2 must raise ValueError; the function raised {type(exc).__name__}")
    pytest.fail("a negative l2 penalty must raise ValueError, not return a fit")
