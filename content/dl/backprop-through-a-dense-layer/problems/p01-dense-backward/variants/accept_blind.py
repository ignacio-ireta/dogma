from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_real_array(name: str, value: object) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype == np.bool_ or not (
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric array")
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    n, m = value.shape
    if not (1 <= n <= 64 and 1 <= m <= 64):
        raise ValueError(f"{name} dimensions must be between 1 and 64")
    arr = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite entries")
    if np.any(np.abs(arr) > 1e6):
        raise ValueError(f"{name} entries must have absolute value <= 1e6")
    return arr


def dense_backward(
    X: NDArray[np.float64],
    W: NDArray[np.float64],
    G: NDArray[np.float64],
    reduction: str = "sum",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return dX, dW, db for Y=X@W+b; mean divides all three by batch size.

    Finite real arrays X:(n,d), W:(d,k), G:(n,k), nonempty dimensions <=64,
    absolute values <=1e6. Cast to float64; reject invalid input with ValueError.
    Never mutate inputs. G is an unreduced upstream gradient.
    """
    if not isinstance(reduction, str) or reduction not in ("sum", "mean"):
        raise ValueError("reduction must be 'sum' or 'mean'")

    Xa = _as_real_array("X", X)
    Wa = _as_real_array("W", W)
    Ga = _as_real_array("G", G)

    n, d = Xa.shape
    if Wa.shape[0] != d:
        raise ValueError("W must have shape (d, k) matching X's second dimension")
    k = Wa.shape[1]
    if Ga.shape != (n, k):
        raise ValueError("G must have shape (n, k)")

    scale = 1.0 if reduction == "sum" else 1.0 / n
    dX = (Ga @ Wa.T) * scale
    dW = (Xa.T @ Ga) * scale
    db = Ga.sum(axis=0) * scale
    return dX.astype(np.float64), dW.astype(np.float64), db.astype(np.float64)
