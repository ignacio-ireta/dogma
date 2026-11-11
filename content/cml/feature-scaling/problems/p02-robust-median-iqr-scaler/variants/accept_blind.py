import numpy as np


def _quantile(sorted_col, q):
    n = len(sorted_col)
    h = q * (n - 1)
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    t = h - lo
    return (1.0 - t) * sorted_col[lo] + t * sorted_col[hi]


def robust_standardize(train, values):
    x = np.asarray(train, dtype=float)
    v = np.asarray(values, dtype=float)
    if x.ndim != 2 or v.ndim != 2:
        raise ValueError("ranks must be two")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("empty training data")
    if v.shape[1] != x.shape[1]:
        raise ValueError("feature count mismatch")
    if not (np.isfinite(x).all() and np.isfinite(v).all()):
        raise ValueError("nonfinite entries")
    center = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    for j in range(x.shape[1]):
        s = np.sort(x[:, j])
        center[j] = _quantile(s, 0.5)
        iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
        scale[j] = 1.0 if iqr < 10 * np.finfo(float).eps else iqr
    return (v - center) / scale, center, scale
