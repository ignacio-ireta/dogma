from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _labels(values: Any, name: str) -> NDArray[np.float64]:
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1-D array of -1 and +1") from exc
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all((arr == 1.0) | (arr == -1.0)):
        raise ValueError(f"{name} entries must be exactly -1 or +1")
    return arr


def adaboost_round(y: ArrayLike, pred: ArrayLike, weights: ArrayLike) -> tuple[float, NDArray[Any]]:
    """Return (alpha, new_weights) for one round of discrete AdaBoost."""
    yv = _labels(y, "y")
    pv = _labels(pred, "pred")
    if yv.shape != pv.shape:
        raise ValueError("y and pred must have the same length")
    try:
        w = np.array(weights, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a 1-D array of numbers") from exc
    if w.ndim != 1 or w.shape != yv.shape:
        raise ValueError("weights must be 1-D with the same length as y")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    peak = float(np.max(w))
    if not peak > 0:
        raise ValueError("weights must have a positive sum")
    scaled = w / peak
    wn = scaled / float(np.sum(scaled))

    wrong = pv != yv
    eps = float(np.sum(wn[wrong]))
    if eps >= 0.5:
        return 0.0, wn
    if eps == 0.0:
        raise ValueError("the classifier is perfect: alpha would be infinite")

    alpha = 0.5 * (math.log1p(-eps) - math.log(eps))
    factor = np.exp(-alpha * yv * pv)
    new = wn * factor
    new = new / float(np.sum(new))
    return float(alpha), new.astype(np.float64)
