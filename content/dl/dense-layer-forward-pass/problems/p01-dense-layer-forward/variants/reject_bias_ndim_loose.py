from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dense_forward(X: ArrayLike, W: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: Y = X @ W + b, with a two-dimensional bias accepted as well.

    Shapes: X (n, d_in), W (d_in, d_out), b (d_out,), Y (n, d_out).
    """
    # Convert once, to float64. Integer inputs must not truncate the result.
    x = np.asarray(X, dtype=np.float64)
    w = np.asarray(W, dtype=np.float64)
    bias = np.asarray(b, dtype=np.float64)

    # Check every shape before computing. A wrong layout can still broadcast or
    # multiply without an error, so the rules are stated and enforced here.
    if x.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d_in), got {x.ndim}-D")
    if w.ndim != 2:
        raise ValueError(f"W must be 2-D (d_in, d_out), got {w.ndim}-D")
    if bias.ndim not in (1, 2):
        raise ValueError(f"b must be 1-D (d_out,), got {bias.ndim}-D")
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"feature mismatch: X has {x.shape[1]} features but W has {w.shape[0]} input rows; "
            "PyTorch stores weights as (d_out, d_in), so transpose them first"
        )
    if bias.shape[0] != w.shape[1]:
        raise ValueError(f"bias has {bias.shape[0]} entries but W has {w.shape[1]} output units")

    # The matrix product sums over the feature axis: Y[i, j] = sum_k X[i, k] * W[k, j].
    # The bias has one entry per output unit, so it broadcasts (n, d_out) + (d_out,):
    # the same bias is added to every row. Cost: n * d_in * d_out multiply-accumulates.
    return x @ w + bias
