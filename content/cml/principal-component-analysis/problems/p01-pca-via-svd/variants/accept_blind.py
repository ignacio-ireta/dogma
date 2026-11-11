import numpy as np


def pca(X, n_components):
    x = np.array(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("X must be 2-D with at least two rows and one column")
    if not np.isfinite(x).all():
        raise ValueError("X must be finite")
    n, d = x.shape
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError("n_components must be an integer")
    k = int(n_components)
    if not 1 <= k <= min(n, d):
        raise ValueError("n_components out of range")
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:k].copy()
    for i in range(k):
        mags = np.abs(comps[i])
        top = mags.max()
        j = int(np.flatnonzero(mags >= top * (1 - 1e-9))[0])
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    ev = s[:k] ** 2 / (n - 1)
    total = float(np.var(x, axis=0, ddof=1).sum())
    ratio = np.zeros(k) if total == 0.0 else ev / total
    return {
        "components": comps,
        "explained_variance": ev,
        "explained_variance_ratio": ratio,
        "transformed": centred @ comps.T,
    }
