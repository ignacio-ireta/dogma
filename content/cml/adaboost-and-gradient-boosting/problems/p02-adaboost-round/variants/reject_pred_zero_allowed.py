from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def adaboost_round(y: ArrayLike, pred: ArrayLike, weights: ArrayLike) -> tuple[float, NDArray[Any]]:
    """Wrong on purpose: return (alpha, new_weights) for one round of discrete AdaBoost.

    The weights are normalised first, so the weighted error is a probability. A classifier at or
    above chance (error >= 1/2) gets alpha = 0 and the round is returned unchanged, so the caller
    can stop. A perfect classifier (error 0) has an infinite alpha, so it is rejected.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(pred, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    if p_arr.shape != y_arr.shape or w_arr.shape != y_arr.shape:
        raise ValueError("y, pred and weights must have the same length")
    if not np.all((y_arr == 1.0) | (y_arr == -1.0)):
        raise ValueError("y must contain only the labels -1 and +1")
    if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
        raise ValueError("weights must be finite and non-negative")
    scale = float(w_arr.max())
    if scale == 0.0:
        raise ValueError("weights must have a positive sum")

    # Rescale before summing so a finite common weight scale cannot overflow the total.
    scaled = w_arr / scale
    w = scaled / scaled.sum()
    error = float(np.sum(w[p_arr != y_arr]))  # weighted share of the mistakes
    if error >= 0.5:
        return 0.0, w.copy()  # no better than chance: no vote, no update
    if error <= 0.0:
        raise ValueError("a perfect classifier has no finite alpha: stop boosting instead")

    alpha = 0.5 * (math.log1p(-error) - math.log(error))
    # Mistakes (y * pred = -1) are scaled up by e^alpha, correct ones down by e^-alpha.
    updated = w * np.exp(-alpha * y_arr * p_arr)
    return float(alpha), updated / updated.sum()
