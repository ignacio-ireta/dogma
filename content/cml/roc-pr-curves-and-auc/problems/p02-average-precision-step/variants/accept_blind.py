from __future__ import annotations

import numpy as np


def average_precision(y_true, scores) -> float:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(labels) != len(values) or len(labels) == 0:
        raise ValueError("lengths differ or input is empty")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    y = labels.astype(int)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("need at least one positive")
    total = 0.0
    previous_recall = 0.0
    for t in sorted(set(values.tolist()), reverse=True):
        flagged = values >= t
        tp = int(np.sum(flagged & (y == 1)))
        precision = tp / int(flagged.sum())
        recall = tp / n_pos
        total += (recall - previous_recall) * precision
        previous_recall = recall
    return float(total)
