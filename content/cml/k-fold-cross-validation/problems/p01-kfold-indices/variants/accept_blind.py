from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray


def kfold_indices(
    n: int,
    k: int,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Return ``k`` (train_idx, test_idx) pairs that partition the positions ``0..n-1``.

    Parameters
    ----------
    n : number of samples.
    k : number of folds, with ``2 <= k <= n``.
    shuffle : if True, test blocks are consecutive chunks of
        ``default_rng(seed).permutation(n)``.
    seed : int, required when ``shuffle`` is True, ignored otherwise.

    Returns
    -------
    A list of ``k`` pairs in fold order. The first ``n % k`` folds have one extra test position.
    Each index array is 1-D, integer typed and sorted ascending. ``train_idx`` is the complement of
    ``test_idx`` in ``range(n)``.

    Raises
    ------
    ValueError
        If ``k < 2``, if ``k > n``, or if ``shuffle`` is True and ``seed`` is None.
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise ValueError("n must be an integer")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError("k must be an integer")
    n, k = int(n), int(k)
    if k < 2:
        raise ValueError("k must be at least 2")
    if k > n:
        raise ValueError("k must not exceed n")
    if shuffle and seed is None:
        raise ValueError("seed is required when shuffle is True")

    if shuffle:
        order = np.random.default_rng(seed).permutation(n).astype(np.intp)
    else:
        order = np.arange(n, dtype=np.intp)

    q, r = divmod(n, k)
    all_positions = np.arange(n, dtype=np.intp)
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    start = 0
    for j in range(k):
        size = q + 1 if j < r else q
        stop = start + size
        test_idx = np.sort(order[start:stop]).astype(np.intp)
        train_idx = np.setdiff1d(all_positions, test_idx).astype(np.intp)
        folds.append((train_idx, test_idx))
        start = stop
    return folds
