from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def permutation_test_mean_diff(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_permutations: int = 5000,
    seed: int,
) -> float:
    """Return the two-sided permutation p-value for the difference in means of ``a`` and ``b``.

    Parameters
    ----------
    a, b : 1-D array-likes of finite real numbers, each non-empty (ints allowed).
    n_permutations : int >= 1, number of random relabellings of the pooled sample.
    seed : int, seeds ``np.random.default_rng``.

    Returns
    -------
    float in (0, 1]: ``(count + 1) / (n_permutations + 1)``, where ``count`` is the number of
    relabellings with ``|mean(A*) - mean(B*)| >= |mean(a) - mean(b)| * (1 - 1e-9)``.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` is not a non-empty 1-D finite array, or if ``n_permutations < 1``.
    """
    # Float copies: the caller's arrays are never written to.
    a_vals = np.asarray(a, dtype=np.float64)
    b_vals = np.asarray(b, dtype=np.float64)
    for name, vals in (("a", a_vals), ("b", b_vals)):
        if vals.ndim != 1:
            raise ValueError(f"group {name} must be 1-D, got shape {vals.shape}")
        if vals.size == 0:
            raise ValueError("a group is empty")
        if not np.all(np.isfinite(vals)):
            raise ValueError("observations must be finite (no NaN or infinity)")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    # Under the null the labels are exchangeable, so the pooled sample is relabelled at random.
    pooled = np.concatenate([a_vals, b_vals])
    n_a = a_vals.size
    observed = abs(float(a_vals.mean()) - float(b_vals.mean()))
    # Exact ties in the difference can land a rounding error below the observed value. The
    # relative tolerance counts them as at least as extreme, as the null distribution requires.
    threshold = observed * (1.0 - 1e-9)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        difference = abs(float(shuffled[:n_a].mean()) - float(shuffled[n_a:].mean()))
        if difference >= threshold:
            count += 1
    # The +1 on both sides counts the observed labelling as one of the possible outcomes. The
    # p-value is then never zero, and it is valid for any finite number of relabellings.
    return (count + 1) / (n_permutations + 1)
