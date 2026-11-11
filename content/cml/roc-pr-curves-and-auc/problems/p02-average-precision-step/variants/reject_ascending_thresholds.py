from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def average_precision(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Wrong on purpose: step-wise average precision: sum over thresholds of (R_k - R_{k-1}) * P_k.

    Thresholds are the distinct scores, highest first. A sample is predicted positive when its
    score is >= the threshold, and tied samples are never split. R_0 = 0.

    Returns
    -------
    AP as a Python float.

    Raises
    ------
    ValueError
        For unequal lengths, empty input, labels other than 0 and 1, non-finite scores, or input
        with no positive sample.
    """
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if len(labels) != len(values):
        raise ValueError("y_true and scores differ in length")
    if len(labels) == 0:
        raise ValueError("at least one sample is required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite: NaN and infinity have no place in the ranking")

    y = labels.astype(np.int64)  # a copy: the caller's labels are never written
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("average precision needs at least one positive sample")

    # Visit the samples from the highest score down, as the threshold falls.
    order = np.argsort(values, kind="stable")
    s_sorted = values[order]
    y_sorted = y[order]
    tp_prefix = np.cumsum(y_sorted)
    flagged_prefix = np.arange(1, len(y_sorted) + 1)

    # A threshold is only evaluated after its whole tie group has been flagged.
    group_ends = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp = tp_prefix[group_ends]
    flagged = flagged_prefix[group_ends]

    precision = tp / flagged  # P_k at each distinct threshold, highest first
    recall = tp / n_pos  # R_k at the same thresholds
    recall_before = np.r_[0.0, recall[:-1]]  # R_{k-1}, with R_0 = 0

    # Each threshold adds a rectangle: the recall it gains, at the precision it reaches.
    return float(np.sum((recall - recall_before) * precision))
