import numbers

import numpy as np


def permutation_test_mean_diff(a, b, *, n_permutations=5000, seed):
    a_vals = np.asarray(a, dtype=np.float64)
    b_vals = np.asarray(b, dtype=np.float64)
    for vals in (a_vals, b_vals):
        if vals.ndim != 1:
            raise ValueError("each group must be one-dimensional")
        if vals.size == 0:
            raise ValueError("a group is empty")
        if not np.all(np.isfinite(vals)):
            raise ValueError("values must be finite")
    if (
        isinstance(n_permutations, bool)
        or not isinstance(n_permutations, numbers.Integral)
        or n_permutations < 1
    ):
        raise ValueError("n_permutations must be an integer of at least 1")
    n_a = a_vals.size
    pooled = np.concatenate([a_vals, b_vals])
    observed = abs(a_vals.mean() - b_vals.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        order = rng.permutation(pooled.size)
        shuffled = pooled[order]
        d = abs(shuffled[:n_a].mean() - shuffled[n_a:].mean())
        # a relabelling counts when it is at least as extreme, up to float rounding
        if d >= observed * (1.0 - 1e-9):
            count += 1
    return (count + 1) / (n_permutations + 1)
