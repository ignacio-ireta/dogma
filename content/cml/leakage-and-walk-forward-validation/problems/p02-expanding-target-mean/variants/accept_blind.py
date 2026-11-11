import numpy as np


def expanding_target_mean(categories, target):
    labels = list(categories)
    y = np.asarray(target, dtype=float)
    if np.asarray(categories, dtype=object).ndim != 1:
        raise ValueError("categories must be 1-D")
    if y.ndim != 1 or len(labels) != len(y):
        raise ValueError("bad shapes")
    if not np.isfinite(y).all():
        raise ValueError("target must be finite")
    seen = {}
    out = np.full(len(labels), np.nan)
    for i, label in enumerate(labels):
        history = seen.get(label, [])
        if history:
            # Divide each term first so that a long run of huge finite targets cannot overflow.
            out[i] = sum(v / len(history) for v in history)
        seen.setdefault(label, []).append(float(y[i]))
    return out
