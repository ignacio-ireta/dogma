from __future__ import annotations

import numbers

import numpy as np


def _prep(X_train, X_query, k):
    Xt = np.asarray(X_train)
    Xq = np.asarray(X_query)
    if Xt.ndim != 2 or Xq.ndim != 2:
        raise ValueError("X arrays must be 2-D")
    if Xt.shape[0] < 1 or Xt.shape[1] < 1:
        raise ValueError("X_train must be non-empty")
    if Xq.shape[1] != Xt.shape[1]:
        raise ValueError("column counts differ")
    try:
        Xt = Xt.astype(np.float64)
        Xq = Xq.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("features must be numeric") from e
    if not (np.all(np.isfinite(Xt)) and np.all(np.isfinite(Xq))):
        raise ValueError("features must be finite")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError("k must be an integer")
    if not 1 <= k <= Xt.shape[0]:
        raise ValueError("k must be between 1 and the number of training rows")
    return Xt, Xq, int(k)


def _dists(Xt, q):
    diff = np.abs(Xt - q)
    r = diff.max(axis=1)
    safe = np.where(r > 0, r, 1.0)
    ssq = ((diff / safe[:, None]) ** 2).sum(axis=1)
    return np.where(r > 0, r * np.sqrt(ssq), 0.0)


def knn_classify(X_train, y_train, X_query, k):
    Xt, Xq, k = _prep(X_train, X_query, k)
    y = np.asarray(y_train)
    if y.ndim != 1 or y.shape[0] != Xt.shape[0]:
        raise ValueError("y_train must be 1-D with one label per training row")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y_train must have an integer dtype")
    y = y.astype(np.int64)
    out = np.empty(Xq.shape[0], dtype=np.int64)
    for i in range(Xq.shape[0]):
        d = _dists(Xt, Xq[i])
        order = np.argsort(d, kind="stable")[:k]
        labs = y[order].tolist()
        counts = {}
        for l in labs:
            counts[l] = counts.get(l, 0) + 1
        best = max(counts.values())
        for l in labs:
            if counts[l] == best:
                out[i] = l
                break
    return out
