from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bootstrap_oob(n: int, n_trees: int, seed: int) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Wrong on purpose: treats seed zero as seed one.

    A bootstrap sample is n draws from the rows 0..n-1 with replacement. The counts of those draws
    are the in-bag weights; rows with count zero were never drawn and are out of bag for that tree.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError("n must be an integer of at least 1")
    if isinstance(n_trees, bool) or not isinstance(n_trees, int | np.integer) or n_trees < 1:
        raise ValueError("n_trees must be an integer of at least 1")
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    n, n_trees = int(n), int(n_trees)
    rng = np.random.default_rng(seed or 1)  # one generator for all trees, consumed in tree order
    in_bag = np.zeros((n_trees, n), dtype=np.int64)
    for t in range(n_trees):
        draws = rng.integers(0, n, size=n)  # n indices, each in [0, n): with replacement
        in_bag[t] = np.bincount(draws, minlength=n)  # how often each row was drawn
    oob = in_bag == 0  # never drawn, so out of bag for this tree
    return in_bag, oob
