import numbers

import numpy as np


def bootstrap_percentile_ci(x, statistic, *, n_resamples=2000, confidence=0.95, seed):
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("the sample must be one-dimensional")
    if values.size == 0:
        raise ValueError("the sample is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("the sample must be finite")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, numbers.Integral)
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be an integer of at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    n = values.size
    rng = np.random.default_rng(seed)
    # row b of the index matrix is resample b; every row is one draw with replacement
    index = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.array([statistic(values[row]) for row in index])
    alpha = 1.0 - confidence
    lo, hi = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(lo), float(hi)
