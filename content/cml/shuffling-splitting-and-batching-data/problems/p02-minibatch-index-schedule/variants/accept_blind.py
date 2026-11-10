from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def minibatch_indices(
    n: int,
    batch_size: int,
    epochs: int,
    *,
    seed: int,
    drop_last: bool = False,
) -> list[list[NDArray[np.int64]]]:
    """Return the batches of each epoch as index arrays, following problem.md.

    The result has one inner list per epoch. Each inner list holds consecutive slices of that
    epoch's permutation of 0..n-1, each slice of batch_size rows except possibly the last.
    One generator is created from seed for the whole call and draws one permutation per epoch.
    Raises ValueError when n < 1, batch_size is outside 1..n, or epochs is negative.
    """
    for value in (n, batch_size, epochs):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError("n, batch_size and epochs must be integers")
    n, batch_size, epochs = int(n), int(batch_size), int(epochs)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not (1 <= batch_size <= n):
        raise ValueError("batch_size must be an integer in 1..n")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    rng = np.random.default_rng(seed)
    schedule: list[list[NDArray[np.int64]]] = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        batches: list[NDArray[np.int64]] = []
        for start in range(0, n, batch_size):
            batch = perm[start : start + batch_size].astype(np.int64)
            if drop_last and len(batch) < batch_size:
                break
            batches.append(batch)
        schedule.append(batches)
    return schedule
