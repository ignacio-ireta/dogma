import numpy as np


def _median(values):
    s = np.sort(values)
    n = s.size
    if n % 2:
        return s[n // 2]
    # Halve before adding so two large finite middle values cannot overflow.
    return s[n // 2 - 1] / 2.0 + s[n // 2] / 2.0


def robust_zscore_outliers(x, threshold=3.5):
    v = np.asarray(x, dtype=float)
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
        raise ValueError("bad input")
    if isinstance(threshold, bool) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("bad threshold")
    med = _median(v)
    half_dev = v / 2.0 - med / 2.0  # the deviation, at half scale, cannot overflow
    mad_half = _median(np.abs(half_dev))  # the MAD at the same half scale
    if mad_half == 0:
        return v != med
    # The ratio of deviation to MAD is the same at half scale.
    return np.abs(0.6745 * half_dev / mad_half) > threshold
