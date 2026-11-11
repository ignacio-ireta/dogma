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
    if initial_train < 1 or horizon < 1 or step < 1 or gap < 0:
        raise ValueError("invalid window parameters")
    if n < initial_train + gap + horizon:
        raise ValueError("no split fits")
    out = []
    c = initial_train
    while c + gap + horizon <= n:
        start = 0 if expanding else c - initial_train
        train = np.arange(start, c, dtype=np.intp)
        test = np.arange(c + gap, c + gap + horizon, dtype=np.intp)
        out.append((train, test))
        c += step
    return out
