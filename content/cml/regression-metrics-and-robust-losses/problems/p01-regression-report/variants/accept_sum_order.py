from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return the MAE, MSE, RMSE and R^2 of y_pred against y_true, as a dict of floats.

    Keys: "mae", "mse", "rmse", "r2". R^2 = 1 - SS_res / SS_tot, with SS_tot taken around the
    mean of y_true, and it is not clipped. Raises ValueError for inputs that are not 1-D, differ in
    length, are empty or non-finite, or whose y_true values are all equal (R^2 undefined).
    Inputs must not be mutated.
    """
    # np.array always copies, so the caller's arrays are never written to. Integer input is
    # converted to float64 here, which avoids integer arithmetic in later reductions.
    truth = np.array(y_true, dtype=np.float64)
    pred = np.array(y_pred, dtype=np.float64)
    if truth.ndim != 1 or pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D, got shapes {truth.shape} and {pred.shape}"
        )
    if truth.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {truth.size} and {pred.size}"
        )
    if truth.size == 0:
        raise ValueError("need at least one sample to score")
    if not (np.all(np.isfinite(truth)) and np.all(np.isfinite(pred))):
        raise ValueError("y_true and y_pred must contain only finite numbers")
    # SS_tot is zero exactly when every y_true value is equal. Check before dividing by it.
    if np.all(truth == truth[0]):
        raise ValueError("R^2 is undefined when every y_true value is equal (zero variance)")

    # Residuals e_i = y_i - yhat_i. The three error scores are summaries of |e| or e^2.
    residual = truth - pred
    mae = np.sum(np.abs(residual)[::-1]) / truth.size
    mse = np.mean(residual**2)
    rmse = np.sqrt(mse)  # the root puts MSE back in the units of y

    # R^2 compares the squared error of the model with the squared error of predicting the mean
    # of y_true. The denominator centres y_true at its own mean, not at the mean of y_pred.
    # A nonconstant target can have deviations whose squares underflow to zero. Rescale
    # both sums by the same deviation scale before squaring; the ratio is unchanged.
    deviation = truth - truth.mean()
    scale = np.max(np.abs(deviation))
    ss_res = np.sum((residual / scale) ** 2)
    ss_tot = np.sum((deviation / scale) ** 2)
    r2 = 1.0 - ss_res / ss_tot  # not clipped: worse-than-mean predictions give a negative value

    return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}
