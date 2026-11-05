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
    """Return walk-forward ``(train_idx, test_idx)`` splits over the rows ``0..n-1`` in time order.

    Parameters
    ----------
    n : number of rows, in time order.
    initial_train : size of the first training window; with ``expanding=False`` every
        window has this size.
    horizon : rows in each test window (at least 1).
    step : how far the training cutoff moves between consecutive splits (at least 1).
    gap : rows skipped between the last training row and the first test row (at least 0).
    expanding : True for expanding windows, False for rolling windows of fixed length.

    Returns
    -------
    Split ``j`` has cutoff ``c_j = initial_train + j * step``, training rows ``[0, c_j)``
    (expanding) or ``[c_j - initial_train, c_j)`` (rolling), and test rows
    ``[c_j + gap, c_j + gap + horizon)``.
    Splits are generated while ``c_j + gap + horizon <= n``.

    Raises
    ------
    ValueError
        If ``initial_train``, ``horizon`` or ``step`` is below 1, if ``gap`` is negative, or if
        ``n < initial_train + gap + horizon``.
    """
    raise NotImplementedError
