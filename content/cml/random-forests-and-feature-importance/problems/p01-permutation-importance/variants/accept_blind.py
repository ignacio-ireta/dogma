import numpy as np


def permutation_importance(predict, X, y, metric, *, n_repeats, seed):
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ValueError("n_repeats must be an integer of at least 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    X = np.array(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a non-empty finite 2-D array")
    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1-D with one target per row of X")

    def score(Z):
        pred = np.asarray(predict(Z))
        if pred.shape != (Z.shape[0],):
            raise ValueError("predict must return one prediction per row")
        return float(metric(y, pred))

    n, d = X.shape
    baseline = score(X)
    rng = np.random.default_rng(seed)
    drops = np.zeros((d, int(n_repeats)))
    for j in range(d):
        for r in range(int(n_repeats)):
            perm = rng.permutation(n)
            Xp = X.copy()
            Xp[:, j] = X[perm, j]
            drops[j, r] = baseline - score(Xp)
    return drops.mean(axis=1), drops.std(axis=1)
