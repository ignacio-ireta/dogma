from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def early_stopping(
    val_losses: ArrayLike | Sequence[float],
    *,
    patience: int,
    min_delta: float = 0.0,
) -> tuple[int, int]:
    """Return ``(best_epoch, stop_epoch)`` for a sequence of validation losses, one per epoch."""
    losses = np.asarray(val_losses, dtype=float)
    if losses.ndim != 1:
        raise ValueError("val_losses must be 1-D")
    if losses.size == 0:
        raise ValueError("val_losses must be non-empty")
    if not np.all(np.isfinite(losses)):
        raise ValueError("val_losses must be finite")
    if isinstance(patience, bool) or int(patience) != patience or patience < 1:
        raise ValueError("patience must be an integer >= 1")
    min_delta = float(min_delta)
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative")
    patience = int(patience)

    best = math.inf
    best_epoch = 0
    wait = 0
    last = len(losses) - 1
    for epoch in range(len(losses)):
        loss = float(losses[epoch])
        if loss < best - min_delta:
            best = loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            return best_epoch, epoch
    return best_epoch, last
