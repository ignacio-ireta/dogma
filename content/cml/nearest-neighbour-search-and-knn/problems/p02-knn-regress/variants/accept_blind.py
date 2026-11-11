from __future__ import annotations

import numbers

import numpy as np


def _dists(Xt, q):
    diff = np.abs(Xt - q)
    r = diff.max(axis=1)
    safe = np.where(r > 0, r, 1.0)
    ssq = ((diff / safe[:, None]) ** 2).sum(axis=1)
    return np.where(r > 0, r * np.sqrt(ssq), 0.0)


def knn_regress(X_train, y_train, X_query, k, *, weights="uniform"):  # noqa: PLR0912
    if not isinstance(weights, str) or weights not in ("uniform", "distance"):
        raise ValueError('weights must be "uniform" or "distance"')
    Xt = np.asarray(X_train)
    Xq = np.asarray(X_query)
    y = np.asarray(y_train)
    if Xt.ndim != 2 or Xq.ndim != 2 or y.ndim != 1:
        raise ValueError("bad shapes")
    if Xt.shape[0] < 1 or Xt.shape[1] < 1:
        raise ValueError("X_train must be non-empty")
    if Xq.shape[1] != Xt.shape[1]:
        raise ValueError("column counts differ")
    if y.shape[0] != Xt.shape[0]:
        raise ValueError("length mismatch")
    try:
        Xt = Xt.astype(np.float64)
        Xq = Xq.astype(np.float64)
        y = y.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("values must be numeric") from e
    if not (np.all(np.isfinite(Xt)) and np.all(np.isfinite(Xq)) and np.all(np.isfinite(y))):
        raise ValueError("values must be finite")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError("k must be an integer")
    if not 1 <= k <= Xt.shape[0]:
        raise ValueError("k must be between 1 and the number of training rows")
    k = int(k)
    out = np.empty(Xq.shape[0], dtype=np.float64)
    for i in range(Xq.shape[0]):
        d = _dists(Xt, Xq[i])
        order = np.argsort(d, kind="stable")[:k]
        dn, yn = d[order], y[order]
        if weights == "uniform":
            out[i] = yn.mean()
        else:
            zero = dn == 0
            if zero.any():
                out[i] = yn[zero].mean()
            else:
                s = dn.min()
                w = 1.0 / (dn / s)
                out[i] = (w * yn).sum() / w.sum()
    return out
