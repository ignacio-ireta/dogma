from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dense_forward(X: ArrayLike, W: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return Y = X @ W + b for one dense layer applied to a batch.

    Parameters
    ----------
    X : (n, d_in) array-like of real numbers, one example per row. n and d_in may be 0.
    W : (d_in, d_out) array-like. Column j holds the weights of output unit j. This is the
        transpose of PyTorch's nn.Linear storage, which is (d_out, d_in).
    b : (d_out,) array-like, exactly one-dimensional. It is added to every row.

    Returns
    -------
    (n, d_out) float64 array with Y[i] = X[i] @ W + b.

    Raises
    ------
    ValueError
        If X or W is not 2-D, if b is not 1-D, if X has a different number of features
        than W has input rows, or if b's length differs from W's number of columns.
    """
    raise NotImplementedError
