from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def bias_variance(
    predictions: ArrayLike,
    f_true: ArrayLike,
    noise_var: float,
) -> tuple[float, float, float]:
    """Return ``(bias2, variance, expected_mse)`` averaged over the test points."""
    preds = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(f_true, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError("predictions must be 2-D with shape (B, m)")
    if preds.shape[0] < 1 or preds.shape[1] < 1:
        raise ValueError("predictions needs at least one model and one test point")
    if truth.ndim != 1 or truth.shape[0] != preds.shape[1]:
        raise ValueError("f_true must be 1-D with one value per test point")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(truth)):
        raise ValueError("predictions and f_true must be finite")
    noise = float(noise_var)
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise_var must be finite and non-negative")

    # Work with the deviations from the truth so a large common offset keeps its precision.
    dev = preds - truth
    mean_dev = dev.mean(axis=0)
    bias2_j = mean_dev * mean_dev
    var_j = ((dev - mean_dev) ** 2).mean(axis=0)

    bias2 = float(np.mean(bias2_j))
    variance = float(np.mean(var_j))
    expected = float(np.mean(bias2_j + var_j)) + noise
    return bias2, variance, expected
