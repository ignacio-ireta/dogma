from __future__ import annotations

from numpy.typing import ArrayLike


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return the MAE, MSE, RMSE and R^2 of y_pred against y_true, as a dict of floats.

    Keys: "mae", "mse", "rmse", "r2". R^2 = 1 - SS_res / SS_tot, with SS_tot taken around the
    mean of y_true, and it is not clipped. Raises ValueError for inputs that are not 1-D, differ in
    length, are empty or non-finite, or whose y_true values are all equal (R^2 undefined).
    Inputs must not be mutated.
    """
    raise NotImplementedError
