from __future__ import annotations

import math

import numpy as np


def _arr(a):
    x = np.asarray(a)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("input must be a non-empty 1-D sequence")
    try:
        x = x.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("input must be numeric") from e
    if not np.all(np.isfinite(x)):
        raise ValueError("input must be finite")
    return x


def ks_drift(reference, current, *, alpha=0.05):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    a = np.sort(_arr(reference))
    b = np.sort(_arr(current))
    n, m = a.size, b.size
    pooled = np.concatenate([a, b])
    fa = np.searchsorted(a, pooled, side="right") / n
    fb = np.searchsorted(b, pooled, side="right") / m
    d = float(np.max(np.abs(fa - fb)))
    lam = math.sqrt(n * m / (n + m)) * d
    if lam < 0.2:
        p = 1.0
    else:
        p = 0.0
        for k in range(1, 200):
            term = math.exp(-2.0 * k * k * lam * lam)
            p += (-1) ** (k - 1) * term
            if term < 1e-18:
                break
        p = min(1.0, max(0.0, 2.0 * p))
    return d, p, bool(p < alpha)
