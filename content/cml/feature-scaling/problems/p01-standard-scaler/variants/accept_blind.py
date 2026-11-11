from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_BOUND = 1e100


def _as_matrix(a: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        arr = np.array(a, dtype=np.float64)
    except (TypeError, ValueError, np.exceptions.ComplexWarning) as exc:
        raise ValueError(f"{name} must be real numeric") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if not np.all(np.isfinite(arr)) or np.any(np.abs(arr) > _BOUND):
        raise ValueError(f"{name} must be finite with |entries| <= 1e100")
    return arr


def standardize(
    train: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = _as_matrix(train, "train")
    v = _as_matrix(values, "values")
    n, d = x.shape
    if n < 1 or d < 1:
        raise ValueError("train must be non-empty")
    if v.shape[1] != d:
        raise ValueError("feature counts differ")
    x0 = x[0]
    constant = np.all(x == x0, axis=0)
    mean = x0 + np.mean(x - x0, axis=0)
    mean = np.where(constant, x0, mean)
    dev = x - mean
    m = np.max(np.abs(dev), axis=0)
    safe = np.where(m > 0, m, 1.0)
    scale = safe * np.sqrt(np.mean((dev / safe) ** 2, axis=0))
    scale = np.where(constant, 1.0, scale)
    scale = np.where(~constant & (scale == 0), np.finfo(np.float64).tiny * 0 + m, scale)
    out = (v - mean) / scale
    return out.astype(np.float64), mean.astype(np.float64), scale.astype(np.float64)
