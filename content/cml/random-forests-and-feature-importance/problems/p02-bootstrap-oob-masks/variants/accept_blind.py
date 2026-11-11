from __future__ import annotations

import numbers

import numpy as np


def _isint(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def bootstrap_oob(n, n_trees, seed):
    if not _isint(n) or n < 1:
        raise ValueError("n must be an integer >= 1")
    if not _isint(n_trees) or n_trees < 1:
        raise ValueError("n_trees must be an integer >= 1")
    if not _isint(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    n, n_trees, seed = int(n), int(n_trees), int(seed)
    rng = np.random.default_rng(seed)
    in_bag = np.zeros((n_trees, n), dtype=np.int64)
    for t in range(n_trees):
        idx = rng.integers(0, n, size=n)
        in_bag[t] = np.bincount(idx, minlength=n)
    return in_bag, in_bag == 0
