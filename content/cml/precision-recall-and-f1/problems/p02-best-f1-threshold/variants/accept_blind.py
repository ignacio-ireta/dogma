from __future__ import annotations

import numpy as np


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(labels) != len(values) or len(labels) == 0:
        raise ValueError("lengths differ or input is empty")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    y = labels.astype(int).tolist()
    total_pos = sum(y)
    pairs = sorted(zip(values.tolist(), y), key=lambda p: -p[0])  # noqa: B905
    best_f1, best_t = -1.0, None
    tp = fp = 0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == t:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        fn = total_pos - tp
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        if f1 > best_f1:  # thresholds run from high to low: the first best is the largest
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)
