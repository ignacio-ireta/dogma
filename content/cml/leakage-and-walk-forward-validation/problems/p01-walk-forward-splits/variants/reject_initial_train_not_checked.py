from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def walk_forward_splits(
    n: int,
    *,
    initial_train: int,
    horizon: int,
    step: int,
    gap: int = 0,
    expanding: bool = True,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Wrong on purpose: return walk-forward (train_idx, test_idx) splits over the rows ``0..n-1`` in time order.

    Parameters
    ----------
    n : number of rows, in time order.
    initial_train : size of the first training window. Expanding: later windows grow
        from row 0. Rolling: every window has exactly this many rows.
    horizon : number of rows in each test window (at least 1).
    step : how far the training cutoff moves between consecutive splits (at least 1).
    gap : rows skipped between the last training row and the first test row (at least 0).
    expanding : True for expanding windows, False for rolling windows of fixed length.

    Returns
    -------
    Split ``j`` has cutoff ``c_j = initial_train + j * step``. Its training rows are
    ``[0, c_j)`` when expanding, or ``[c_j - initial_train, c_j)`` when rolling. Its test rows
    are ``[c_j + gap, c_j + gap + horizon)``. Splits are generated while the test window ends at or
    before ``n``.

    Raises
    ------
    ValueError
        If ``initial_train``, ``horizon`` or ``step`` is below 1, if ``gap`` is negative, or if
        ``n < initial_train + gap + horizon`` (no split fits).
    """  # noqa: E501
    if horizon < 1:
        raise ValueError("horizon must be at least 1 row")
    if step < 1:
        raise ValueError("step must be at least 1 row")
    if gap < 0:
        raise ValueError("gap cannot be negative")
    if n < initial_train + gap + horizon:
        raise ValueError("no split fits: n must be at least initial_train + gap + horizon")

    # Step 1: walk the cutoff forward by `step`, starting at initial_train.
    # Step 2: each split trains strictly before its cutoff and tests `gap` rows after it.
    # Step 3: stop as soon as the next test window would run past the last row.
    splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    cut = initial_train
    while cut + gap + horizon <= n:
        train_start = 0 if expanding else cut - initial_train
        train = np.arange(train_start, cut, dtype=np.intp)
        test = np.arange(cut + gap, cut + gap + horizon, dtype=np.intp)
        splits.append((train, test))
        cut += step
    return splits
