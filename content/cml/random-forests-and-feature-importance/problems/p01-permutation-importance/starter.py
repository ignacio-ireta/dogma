from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def permutation_importance(
    predict: Callable[[NDArray[np.float64]], ArrayLike],
    X: ArrayLike,
    y: ArrayLike,
    metric: Callable[[Any, Any], float],
    *,
    n_repeats: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (mean_drop, std_drop) per feature, by permuting one column at a time.

    One generator, np.random.default_rng(seed), is used throughout. For each feature j in order,
    and each repeat in order, draw perm = rng.permutation(n), replace column j of a copy of X with
    X[perm, j], and record baseline - metric(y, predict(X_perm)). std_drop uses ddof = 0.
    Raises ValueError for invalid X, y, n_repeats, seed, or a predict output of the wrong length.
    Must not mutate X or y.
    """
    raise NotImplementedError
