from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bootstrap_oob(n: int, n_trees: int, seed: int) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Return (in_bag, oob) for n_trees bootstrap samples of n rows, drawn with replacement.

    One generator, np.random.default_rng(seed), serves all trees in order. Tree t draws
    rng.integers(0, n, size=n). in_bag[t, i] counts how often row i was drawn; oob[t, i] is True
    where in_bag[t, i] == 0. Raises ValueError unless n and n_trees are integers >= 1 and seed is
    a non-negative integer.
    """
    raise NotImplementedError
