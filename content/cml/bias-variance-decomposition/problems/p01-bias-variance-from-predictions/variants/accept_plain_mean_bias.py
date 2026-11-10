from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def bias_variance(
    predictions: ArrayLike,
    f_true: ArrayLike,
    noise_var: float,
) -> tuple[float, float, float]:
    """Return ``(bias2, variance, expected_mse)`` averaged over the test points.

    Parameters
    ----------
    predictions : 2-D array-like, shape ``(B, m)``: row ``b`` holds model ``b``'s predictions at the
        ``m`` test points. Each model was trained on its own training set from the same process.
    f_true : 1-D array-like of length ``m``, the noise-free target at each test point.
    noise_var : the variance of the label noise, so that a test label is ``f_true + noise``.

    Returns
    -------
    bias2 : mean over test points of ``(mean_b predictions - f_true) ** 2``.
    variance : mean over test points of the variance of the predictions over models (divisor ``B``).
    expected_mse : mean over test points of ``bias2_j + variance_j + noise_var``, which equals
        ``bias2 + variance + noise_var``. Preserve representable errors at large common offsets;
        squared quantities outside float64 range may overflow to infinity or underflow to zero.

    Raises
    ------
    ValueError
        If ``predictions`` is not 2-D or has no model or no test point, if ``f_true`` is not 1-D
        or does not match the number of test points, if any value is non-finite, or if
        ``noise_var`` is negative or non-finite.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(f_true, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError("predictions must be 2-D: one row per model, one column per test point")
    n_models, n_points = preds.shape
    if n_models < 1 or n_points < 1:
        raise ValueError("predictions must hold at least one model and one test point")
    if truth.ndim != 1 or truth.shape[0] != n_points:
        raise ValueError("f_true must be 1-D with one value per test point")
    if not (np.all(np.isfinite(preds)) and np.all(np.isfinite(truth))):
        raise ValueError("predictions and f_true must be finite")
    noise = float(noise_var)
    if not (np.isfinite(noise) and noise >= 0.0):
        raise ValueError("noise_var must be a finite, non-negative number")

    # Work around a per-point anchor before averaging. A large shared prediction
    # offset must not overflow a sum or erase the much smaller model deviations.
    bias_terms = np.empty(n_points, dtype=np.float64)
    variance_terms = np.empty(n_points, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for point in range(n_points):
            column = preds[:, point]
            offsets = column - column[0]
            if np.all(np.isfinite(offsets)):
                offset_mean = _stable_mean(offsets)
                bias_error = (column[0] - truth[point]) + offset_mean
                variance_terms[point] = _mean_square(offsets - offset_mean)
            else:
                # Opposite extreme predictions can overflow their difference. The
                # normalized mean stays representable; their variance exceeds float64
                # range for every array size NumPy can represent.
                magnitude = float(np.max(np.abs(column)))
                mean_prediction = magnitude * float(np.mean(column / magnitude))
                bias_error = mean_prediction - truth[point]
                variance_terms[point] = np.inf
            bias_terms[point] = bias_error * bias_error

    bias2 = float(np.mean(bias_terms))
    variance = _stable_mean(variance_terms)
    return bias2, variance, float(bias2 + variance + noise)


def _stable_mean(values: np.ndarray) -> float:
    """Average finite values without overflowing their sum; preserve zero and infinity."""
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0 or np.isinf(magnitude):
        return magnitude
    return float(magnitude * np.mean(values / magnitude))


def _mean_square(values: np.ndarray) -> float:
    """Normalize before squaring, then restore scale after the square root."""
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        return 0.0
    root_mean_square = magnitude * float(np.sqrt(np.mean((values / magnitude) ** 2)))
    return float(root_mean_square * root_mean_square)
