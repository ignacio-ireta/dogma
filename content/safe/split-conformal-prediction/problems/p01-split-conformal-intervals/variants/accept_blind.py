from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


def _arr(x, allow_empty):
    try:
        a = np.asarray(x)
    except Exception as e:
        raise ValueError("bad array") from e
    if a.dtype == bool or a.dtype.kind not in "iuf":
        raise ValueError("non-numeric or complex input")
    if a.ndim != 1:
        raise ValueError("must be 1-D")
    if a.size == 0 and not allow_empty:
        raise ValueError("empty")
    a = a.astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite")
    return a


def split_conformal_interval(
    cal_pred: NDArray[np.float64],
    cal_true: NDArray[np.float64],
    test_pred: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(alpha, bool) or not isinstance(alpha, (float, np.floating)):
        raise ValueError("alpha must be a float")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha out of range")
    cp = _arr(cal_pred, False)
    ct = _arr(cal_true, False)
    tp = _arr(test_pred, True)
    if cp.shape != ct.shape:
        raise ValueError("length mismatch")
    n = cp.size
    scores = np.sort(np.abs(ct - cp))
    k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
    k = max(k, 1)
    q = float(scores[k - 1]) if k <= n else math.inf
    return tp - q, tp + q
