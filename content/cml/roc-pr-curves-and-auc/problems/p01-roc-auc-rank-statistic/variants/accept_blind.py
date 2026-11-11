from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def roc_auc(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(labels) != len(values):
        raise ValueError("lengths differ")
    if len(labels) == 0:
        raise ValueError("empty input")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    y = labels.astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes")
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        midrank = (start + 1 + end + 1) / 2  # 1-based positions start+1 .. end+1
        for k in range(start, end + 1):
            ranks[order[k]] = midrank
        start = end + 1
    r_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
