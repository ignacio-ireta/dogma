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
    raise NotImplementedError
