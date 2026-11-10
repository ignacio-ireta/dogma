from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _matrix(value: NDArray[np.float64], limit: float, size: int) -> NDArray[np.float64]:
    """Validate before casting so complex data cannot silently lose information."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("expected real numeric arrays")
    array = raw.astype(np.float64)
    if array.ndim != 2 or not all(1 <= n <= size for n in array.shape):
        raise ValueError("expected a nonempty bounded matrix")
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > limit):
        raise ValueError("values outside finite bounds")
    return array


def dense_backward(
    X: NDArray[np.float64],
    W: NDArray[np.float64],
    G: NDArray[np.float64],
    reduction: str = "sum",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: return dX:(n,d), dW:(d,k), db:(k,) for Y=X@W+b.

    X:(n,d), W:(d,k), G:(n,k) are finite real matrices, dimensions 1..64,
    absolute values <=1e6. Cast to float64; invalid input raises ValueError.
    G is unreduced; mean divides every result by n. Inputs are never mutated.
    """
    X, W, G = (_matrix(a, 1e6, 64) for a in (X, W, G))
    if X.shape[1] != W.shape[0] or G.shape != (X.shape[0], W.shape[1]):
        raise ValueError("incompatible dense-layer shapes")
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be sum or mean")
    scale = X.shape[0] if reduction == "mean" else 1
    # A single upstream signal fans out to inputs, weights and broadcast bias.
    upstream = G / scale
    return upstream @ W.T, X.T @ upstream, upstream.sum(axis=0).astype(np.float32)
