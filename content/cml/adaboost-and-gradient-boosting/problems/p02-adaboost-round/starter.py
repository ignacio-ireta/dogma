from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike, NDArray


def adaboost_round(y: ArrayLike, pred: ArrayLike, weights: ArrayLike) -> tuple[float, NDArray[Any]]:
    """Return (alpha, new_weights) for one round of discrete AdaBoost.

    y, pred : labels in {-1, +1} (same length n >= 1). weights : finite, non-negative, positive sum.
    Normalise the weights to sum 1, then eps = sum of w_i where pred_i != y_i.
    If eps >= 0.5 return (0.0, normalised weights). If eps == 0 raise ValueError.
    Otherwise alpha = 0.5 * ln((1 - eps) / eps) and new weights are
    w_i * exp(-alpha * y_i * pred_i), normalised to sum 1. Returns a Python float and a
    float64 array. Must not mutate inputs.
    """
    raise NotImplementedError
