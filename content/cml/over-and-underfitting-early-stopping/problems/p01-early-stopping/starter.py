from __future__ import annotations

from collections.abc import Sequence

from numpy.typing import ArrayLike


def early_stopping(
    val_losses: ArrayLike | Sequence[float],
    *,
    patience: int,
    min_delta: float = 0.0,
) -> tuple[int, int]:
    """Return ``(best_epoch, stop_epoch)`` for a sequence of validation losses, one per epoch.

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
    """
    raise NotImplementedError
