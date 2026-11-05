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
    """Return the batches of each epoch as index arrays, following problem.md.

    The result has one inner list per epoch. Each inner list holds consecutive slices of that
    epoch's permutation of 0..n-1, each slice of batch_size rows except possibly the last.
    One generator is created from seed for the whole call and draws one permutation per epoch.
    Raises ValueError when n < 1, batch_size is outside 1..n, or epochs is negative.
    """
    raise NotImplementedError
