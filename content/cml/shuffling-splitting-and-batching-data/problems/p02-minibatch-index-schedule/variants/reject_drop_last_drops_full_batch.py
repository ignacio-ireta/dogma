from __future__ import annotations

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
    """Wrong on purpose: return the batches of each epoch as index arrays, following problem.md.

    The result has one inner list per epoch. Each inner list holds consecutive slices of that
    epoch's permutation of 0..n-1, each slice of batch_size rows except possibly the last.
    One generator is created from seed for the whole call and draws one permutation per epoch.
    Raises ValueError when n < 1, batch_size is outside 1..n, or epochs is negative.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"n must be an integer of at least 1, got {n!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int | np.integer):
        raise ValueError(f"batch_size must be an integer, got {batch_size!r}")
    if not 1 <= batch_size <= n:
        raise ValueError(f"batch_size must be between 1 and {n}, got {batch_size}")
    if isinstance(epochs, bool) or not isinstance(epochs, int | np.integer) or epochs < 0:
        raise ValueError(f"epochs must be a non-negative integer, got {epochs!r}")

    # One generator for the whole run. Creating it inside the epoch loop would repeat the shuffle.
    rng = np.random.default_rng(seed)
    schedule: list[list[NDArray[np.int64]]] = []
    for _ in range(epochs):
        # The one draw of this epoch: a permutation of all row indices.
        perm = rng.permutation(n)
        batches: list[NDArray[np.int64]] = []
        for start in range(0, n, batch_size):
            batch = perm[start : start + batch_size]
            # The final slice may be short. drop_last removes it; otherwise it is kept.
            if drop_last and len(batch) <= batch_size:
                break
            # Keep the permutation order inside each batch. Do not sort it.
            batches.append(batch.astype(np.int64))
        schedule.append(batches)
    return schedule
