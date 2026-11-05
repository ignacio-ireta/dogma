import numpy as np
import pytest
from submission import regression_report


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def test_worked_example_gives_hand_computed_scores():
    report = regression_report([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
    assert set(report) == {"mae", "mse", "rmse", "r2"}, (
        "the report must have exactly the keys mae, mse, rmse and r2"
    )
    np.testing.assert_allclose(
        report["mae"], 0.5, rtol=1e-12, err_msg="MAE is the mean of |residual|: divide by n"
    )
    np.testing.assert_allclose(
        report["mse"], 0.375, rtol=1e-12, err_msg="MSE is the mean of residual squared"
    )
    np.testing.assert_allclose(
        report["rmse"],
        np.sqrt(0.375),
        rtol=1e-12,
        err_msg="RMSE must be the square root of MSE, not MSE itself",
    )
    np.testing.assert_allclose(
        report["r2"],
        1 - 1.5 / 29.1875,
        rtol=1e-12,
        err_msg="R^2 = 1 - SS_res / SS_tot, with SS_tot around the mean of y_true",
    )


def test_r2_is_negative_when_predictions_are_worse_than_the_mean():
    report = regression_report([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert report["r2"] < 0, "R^2 must not be clipped at zero: a worse-than-mean fit is negative"
    np.testing.assert_allclose(report["r2"], -3.0, rtol=1e-12, err_msg="R^2 formula is wrong")


def test_constant_y_true_raises_value_error():
    _expect_value_error(
        lambda: regression_report([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]),
        "every y_true value is equal, so R^2 is undefined: raise ValueError rather than return a "
        "number",
    )
