from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def minibatch_sgd(
    X: ArrayLike,
    y: ArrayLike,
    lr: float,
    batch_size: int,
    epochs: int,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Fit least squares without intercept by mini-batch gradient descent.

    Minimises L(w) = 1/2 * mean((X w - y)^2) with w starting at zeros. Each epoch shuffles the
    rows with rng.permutation(n), then takes consecutive slices of size batch_size (the last
    slice may be smaller). Each slice gives one update w <- w - lr * g, where
    g = X_b^T (X_b w - y_b) / |b|.

    Returns
    -------
    w : (d,) float64 array of the weights after the last epoch.

    Raises
    ------
    ValueError
        If X is not 2-D, y is not 1-D with len(y) == len(X), batch_size is not in [1, n],
        lr is not positive, or epochs is negative.
    """
    raise NotImplementedError
