from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def train_test_split_indices(
    n: int,
    test_size: float,
    *,
    seed: int,
    stratify: ArrayLike | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return sorted (train_idx, test_idx) index arrays for n rows, following problem.md.

    n is an integer of at least 2. test_size is a float strictly between 0 and 1.
    The test count is ceil(test_size * n). With stratify (labels of length n), each
    class gets its own share of the test rows by largest remainders, and each class is
    shuffled with one draw from one generator created from seed.
    Raises ValueError when n or test_size is invalid, stratify has the wrong shape,
    the training side would be empty, or a class would have no training row.
    Must not mutate stratify.
    """
    raise NotImplementedError
