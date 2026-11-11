from __future__ import annotations

import math

import numpy as np


def _sample(name: str, values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numbers")
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def population_stability_index(expected, actual, *, bins=10, eps=1e-6):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least 2")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("eps must be a finite positive number")
    e = _sample("expected", expected)
    a = _sample("actual", actual)

    # Interior edges: quantiles of expected at k / bins, linear interpolation.
    edges = np.quantile(e, [k / bins for k in range(1, bins)])
    # Count of edges strictly below each value. Values on an edge go to the lower bin.
    e_bin = np.searchsorted(edges, e, side="left")
    a_bin = np.searchsorted(edges, a, side="left")
    p = np.bincount(e_bin, minlength=bins) / e.size
    q = np.bincount(a_bin, minlength=bins) / a.size
    p = np.where(p == 0, eps, p)
    q = np.where(q == 0, eps, q)
    return float(np.sum((q - p) * np.log(q / p)))
