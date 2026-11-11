from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def prediction_sets(cal_probs, cal_labels, test_probs, alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be a float in (0, 1)")
    cal = np.asarray(cal_probs)
    if cal.dtype.kind not in "iuf" or cal.ndim != 2 or cal.shape[0] == 0 or cal.shape[1] < 2:
        raise ValueError("cal_probs must have shape (n, K) with n >= 1 and K >= 2")
    if (
        not np.isfinite(cal).all()
        or (cal < 0).any()
        or not np.allclose(cal.sum(axis=1), 1, atol=1e-9)
    ):
        raise ValueError("cal_probs rows must be finite, nonnegative and sum to 1")
    cal = cal.astype(float)
    n, k_classes = cal.shape
    test = np.asarray(test_probs)
    if test.dtype.kind not in "iuf" or test.ndim != 2 or test.shape[1] != k_classes:
        raise ValueError("test_probs must have shape (m, K) with the same K")
    if test.shape[0] and (
        not np.isfinite(test).all()
        or (test < 0).any()
        or not np.allclose(test.sum(axis=1), 1, atol=1e-9)
    ):
        raise ValueError("test_probs rows must be finite, nonnegative and sum to 1")
    labels = np.asarray(cal_labels)
    if (
        labels.dtype.kind not in "iu"
        or labels.shape != (n,)
        or (labels < 0).any()
        or (labels >= k_classes).any()
    ):
        raise ValueError("cal_labels must be integers in [0, K), one per row")
    test = test.astype(float)
    scores = sorted(1.0 - cal[i, int(labels[i])] for i in range(n))
    rank = math.ceil((n + 1) * (1 - Fraction(repr(float(alpha)))))
    q = math.inf if rank > n else scores[rank - 1]
    return np.asarray((1.0 - test) <= q, dtype=bool).reshape(test.shape[0], k_classes)
