from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def kfold_indices(
    n: int,
    k: int,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Wrong on purpose: returns ``k`` (train_idx, test_idx) pairs that partition the positions ``0..n-1``.

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
    """  # noqa: E501
    if k < 2:
        raise ValueError("k must be at least 2 folds")
    if k > n:
        raise ValueError("k must not exceed n: a fold needs at least one test position")
    if shuffle and seed is None:
        raise ValueError("a shuffled split needs an integer seed to be reproducible")

    # Step 1: the order in which positions are handed to the test blocks.
    # Unshuffled, that is 0..n-1. Shuffled, it is one permutation from a seeded generator.
    order = np.arange(n, dtype=np.intp)
    if seed is not None:
        order = np.random.default_rng(seed).permutation(n).astype(np.intp)

    # Step 2: fold sizes. With q, r = divmod(n, k), the first r folds get q + 1 positions.
    # This is the rule np.array_split uses, and sklearn's KFold uses it too.
    q, r = divmod(n, k)
    sizes = [q + 1 if j < r else q for j in range(k)]
    bounds = np.concatenate([[0], np.cumsum(sizes)])

    # Step 3: fold j tests the block order[bounds[j]:bounds[j+1]]; training is everything else.
    # Sorting makes the output canonical: the result does not depend on the order in which the
    # permutation happened to draw the positions.
    everything = np.arange(n, dtype=np.intp)
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    for j in range(k):
        test = np.sort(order[bounds[j] : bounds[j + 1]])
        train = np.setdiff1d(everything, test)  # complement of the test block, ascending
        folds.append((train, test))
    return folds
