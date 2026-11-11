import math

import numpy as np


def _matrix(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(name + " must be 2-D and non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(name + " must be finite")
    return arr


def _labels(X, C):
    out = np.empty(len(X), dtype=np.int64)
    for i, x in enumerate(X):
        best, best_d = 0, None
        for j, c in enumerate(C):
            dd = float(np.sum((x - c) ** 2))
            if best_d is None or dd < best_d:
                best, best_d = j, dd
        out[i] = best
    return out


def kmeans(X, init_centroids, *, max_iter=300, tol=1e-6):
    X = _matrix(X, "X")
    C = _matrix(init_centroids, "init_centroids").copy()
    if C.shape[1] != X.shape[1]:
        raise ValueError("init_centroids must have the same number of columns as X")
    if not 1 <= C.shape[0] <= X.shape[0]:
        raise ValueError("init_centroids must have between 1 and n rows")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")
    if isinstance(tol, bool) or not math.isfinite(float(tol)) or float(tol) < 0:
        raise ValueError("tol must be finite and >= 0")
    tol = float(tol)
    k = C.shape[0]
    n_iter = 0
    for it in range(1, max_iter + 1):
        lab = _labels(X, C)
        new = C.copy()
        for j in range(k):
            members = X[lab == j]
            if len(members):
                new[j] = members.mean(axis=0)
        shift = max(float(np.linalg.norm(new[j] - C[j])) for j in range(k))
        C = new
        n_iter = it
        if shift <= tol:
            break
    lab = _labels(X, C)
    inertia = float(sum(np.sum((X[i] - C[lab[i]]) ** 2) for i in range(len(X))))
    return C, lab, inertia, n_iter
