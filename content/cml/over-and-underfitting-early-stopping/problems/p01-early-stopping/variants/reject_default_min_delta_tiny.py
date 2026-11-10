from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def early_stopping(
    val_losses: ArrayLike | Sequence[float],
    *,
    patience: int,
    min_delta: float = 1e-12,
) -> tuple[int, int]:
    """Wrong on purpose: default min_delta is a tiny positive number instead of 0. Return ``(best_epoch, stop_epoch)`` for a sequence of validation losses, one per epoch.

    Parameters
    ----------
    val_losses : 1-D array-like of finite floats, epoch 0 first.
    patience : int >= 1, the number of consecutive epochs without improvement that stops training.
    min_delta : float >= 0. An epoch improves only when its loss is below the best loss so far by
        more than ``min_delta``; a decrease of exactly ``min_delta`` does not count.

    Returns
    -------
    best_epoch : the epoch of the last improvement before stopping, i.e. the best checkpoint.
    stop_epoch : the epoch at which the wait counter reaches ``patience``. If it never does, the
        last epoch.

    Raises
    ------
    ValueError
        If ``val_losses`` is empty, not 1-D, or non-finite, if ``patience < 1``, or if
        ``min_delta`` is negative or non-finite.
    """  # noqa: E501
    losses = np.asarray(val_losses, dtype=np.float64)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("val_losses must be a non-empty 1-D sequence, one loss per epoch")
    if not np.all(np.isfinite(losses)):
        raise ValueError("validation losses must be finite (no NaN or infinity)")
    if patience < 1:
        raise ValueError("patience must be at least 1 epoch")
    if not (np.isfinite(min_delta) and min_delta >= 0.0):
        raise ValueError("min_delta must be a finite, non-negative number")

    # Improvement: strictly below the best loss so far by more than min_delta. The first epoch
    # always improves because the best starts at +infinity.
    best = np.inf
    best_epoch = 0
    wait = 0
    for epoch, loss in enumerate(losses):
        if loss < best - min_delta:
            best = float(loss)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return best_epoch, epoch
    return best_epoch, int(losses.size - 1)
