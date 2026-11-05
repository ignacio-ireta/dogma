from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mse_sigmoid_grad(w: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Return dL/dw for L(w) = mean((sigmoid(X @ w) - y)**2).

    Parameters
    ----------
    w : (d,) array-like of finite weights.
    X : (n, d) array-like, one sample per row (n >= 1, d >= 1).
    y : (n,) array-like of targets, one per row of X.

    Returns
    -------
    (d,) float64 array equal to (2/n) * X.T @ ((s - y) * s * (1 - s)), where
    s = sigmoid(X @ w). The sigmoid must be stable for large |z|.

    Raises
    ------
    ValueError
        If X is not 2-D, w is not 1-D, y is not 1-D, the lengths do not match, X is empty,
        or any entry is NaN or infinite.
    """
    raise NotImplementedError
