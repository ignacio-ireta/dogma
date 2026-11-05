import numpy as np
import pytest
from submission import boost_stumps_mse

X_EXAMPLE = np.array([[1.0], [2.0], [3.0], [4.0]])
Y_EXAMPLE = np.array([0.0, 0.0, 1.0, 1.0])


def _call(learning_rate=1.0, n_estimators=1):
    """Fit the model; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return boost_stumps_mse(
            X_EXAMPLE, Y_EXAMPLE, n_estimators=n_estimators, learning_rate=learning_rate
        )
    except Exception as exc:
        pytest.fail(
            f"a valid fit must return (predict, train_pred); the call raised "
            f"{type(exc).__name__}: {exc}"
        )


def test_one_round_separates_the_two_groups():
    _, train_pred = _call()
    np.testing.assert_allclose(
        train_pred,
        [0.0, 0.0, 1.0, 1.0],
        atol=1e-12,
        err_msg="one round with learning rate 1 must move each group to its own mean: "
        "start at the mean of y, then add the stump's leaf values",
    )


def test_learning_rate_shrinks_the_step():
    _, half = _call(learning_rate=0.5)
    np.testing.assert_allclose(
        half,
        [0.25, 0.25, 0.75, 0.75],
        atol=1e-12,
        err_msg="a learning rate of 0.5 must halve the step away from the initial mean",
    )


def test_zero_estimators_raises_value_error():
    try:
        boost_stumps_mse(X_EXAMPLE, Y_EXAMPLE, n_estimators=0, learning_rate=0.5)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"zero estimators must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("zero estimators must raise ValueError, not return a model")
