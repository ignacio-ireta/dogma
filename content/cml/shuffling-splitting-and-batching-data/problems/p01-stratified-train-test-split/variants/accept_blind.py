from __future__ import annotations

import math
import numbers

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
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    if isinstance(test_size, bool) or not isinstance(test_size, numbers.Real):
        raise ValueError("test_size must be a float strictly between 0 and 1")
    test_size = float(test_size)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0 and 1")
    n = int(n)

    n_test = math.ceil(test_size * n)
    if n_test >= n:
        raise ValueError("the training side would be empty")

    if stratify is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test]).astype(np.int64)
        train_idx = np.sort(perm[n_test:]).astype(np.int64)
        return train_idx, test_idx

    labels = np.asarray(stratify)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("stratify must be 1-D with length n")

    classes, counts = np.unique(labels, return_counts=True)
    k = len(classes)
    ideal = [test_size * float(c) for c in counts]
    base = [math.floor(v) for v in ideal]
    frac = [ideal[j] - base[j] for j in range(k)]
    extra = n_test - sum(base)
    order = sorted(range(k), key=lambda j: (-frac[j], j))
    t = list(base)
    for j in order[:extra]:
        t[j] += 1

    for j in range(k):
        if t[j] >= counts[j]:
            raise ValueError("a class would have no training row")

    rng = np.random.default_rng(seed)
    test_parts: list[NDArray[np.int64]] = []
    train_parts: list[NDArray[np.int64]] = []
    for j in range(k):
        rows_j = np.flatnonzero(labels == classes[j]).astype(np.int64)
        perm_j = rng.permutation(rows_j)
        test_parts.append(perm_j[: t[j]])
        train_parts.append(perm_j[t[j] :])

    test_idx = np.sort(np.concatenate(test_parts)).astype(np.int64)
    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
    return train_idx, test_idx
