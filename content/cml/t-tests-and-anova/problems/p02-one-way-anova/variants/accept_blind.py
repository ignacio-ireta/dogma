import numpy as np
from scipy import stats


def one_way_anova(*groups):
    if len(groups) < 2:
        raise ValueError("need at least two groups")
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    for g in arrays:
        if g.ndim != 1 or g.size == 0:
            raise ValueError("each group must be a non-empty one-dimensional sample")
        if not np.all(np.isfinite(g)):
            raise ValueError("observations must be finite")
    k = len(arrays)
    total = sum(g.size for g in arrays)
    if total <= k:
        raise ValueError("need more observations than groups")
    grand = np.concatenate(arrays).mean()
    # between: each group mean weighted by its size; within: squared deviations from each group mean
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in arrays)
    ssw = sum(np.sum((g - g.mean()) ** 2) for g in arrays)
    if ssw == 0.0:
        raise ValueError("within-group variation is zero: F is undefined")
    f = (ssb / (k - 1)) / (ssw / (total - k))
    return float(f), float(stats.f.sf(f, k - 1, total - k))
