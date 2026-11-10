from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def train_test_split_indices(
    n: int,
    test_size: float,
    seed: int,
    stratify: ArrayLike | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Wrong on purpose: return sorted (train_idx, test_idx) index arrays for n rows, following problem.md.

    n is an integer of at least 2. test_size is a float strictly between 0 and 1.
    The test count is ceil(test_size * n). With stratify (labels of length n), each
    class gets its own share of the test rows by largest remainders, and each class is
    shuffled with one draw from one generator created from seed.
    Raises ValueError when n or test_size is invalid, stratify has the wrong shape,
    the training side would be empty, or a class would have no training row.
    Must not mutate stratify.
    """  # noqa: E501
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 2:
        raise ValueError(f"n must be an integer of at least 2, got {n!r}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size!r}")

    # Step 1: the total number of test rows. ceil, not round: 2.5 becomes 3.
    # The product is a float64 expression, evaluated exactly as written.
    n_test = math.ceil(test_size * n)

    # One generator for the whole call, created before any draw.
    rng = np.random.default_rng(seed)

    if stratify is None:
        # Step 2 (unstratified): one permutation of all rows, then split it.
        if n_test >= n:
            raise ValueError("the test share leaves no training rows: lower test_size")
        perm = rng.permutation(n)
        return np.sort(perm[n_test:]).astype(np.int64), np.sort(perm[:n_test]).astype(np.int64)

    labels = np.asarray(stratify)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(f"stratify must be 1-D with one label per row ({n}), got {labels.shape}")

    # Step 2 (stratified): classes in ascending order, with their row counts.
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)

    # Step 3: ideal test count per class and its floor. The leftover rows R go to the
    # classes with the largest fractional parts; ties go to the earlier class.
    ideal = test_size * counts
    base = np.floor(ideal).astype(np.int64)
    frac = ideal - base
    leftover = n_test - int(base.sum())
    order = np.lexsort((np.arange(len(counts)), -frac))  # primary: largest frac first
    extra = np.zeros(len(counts), dtype=np.int64)
    extra[order[:leftover]] = 1
    test_counts = base + extra

    # Step 4: every class must keep at least one training row.
    if np.any(test_counts >= counts):
        raise ValueError("a class would have no training rows: lower test_size or merge classes")

    # Step 5: one draw per class, in ascending class order. The first test_counts[j]
    # shuffled rows of class j are test rows; the rest are training rows.
    test_parts: list[NDArray[np.int64]] = []
    train_parts: list[NDArray[np.int64]] = []
    for j in range(len(classes)):
        rows = np.flatnonzero(inverse == j)
        shuffled = rng.permutation(rows)
        test_parts.append(shuffled[: test_counts[j]])
        train_parts.append(shuffled[test_counts[j] :])
    test_idx = np.sort(np.concatenate(test_parts)).astype(np.int64)
    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
    return train_idx, test_idx
