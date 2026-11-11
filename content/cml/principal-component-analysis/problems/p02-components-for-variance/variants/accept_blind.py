import numpy as np


def components_for_variance(X, threshold):
    x = np.array(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("X must be 2-D with at least two rows and one column")
    if not np.isfinite(x).all():
        raise ValueError("X must be finite")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("threshold must be a real number")
    t = float(threshold)
    if not 0.0 < t <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    n = x.shape[0]
    centred = x - x.mean(axis=0)
    _, s, _ = np.linalg.svd(centred, full_matrices=False)
    total = float(np.var(x, axis=0, ddof=1).sum())
    if total == 0.0:
        raise ValueError("every column is constant")
    ratio = (s**2 / (n - 1)) / total
    cum = np.cumsum(ratio)
    for i, c in enumerate(cum):
        if c >= t - 1e-12:
            return i + 1
    return len(cum)
