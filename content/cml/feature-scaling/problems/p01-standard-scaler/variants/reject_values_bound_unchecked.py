from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def standardize(
    train: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: checks finiteness and magnitude only for training entries.

    train is (n, d), n >= 1, d >= 1; values is (m, d), m >= 0.
    Real numeric entries must be finite with absolute value <= 1e100.
    Use ddof=0 and replace exactly zero scales by 1. Never mutate inputs.
    Raise ValueError for invalid rank, shape, range or nonfinite values.
    Outputs are float64 arrays of shapes (m, d), (d,), (d,).
    """
    fitted = np.asarray(train, dtype=np.float64)
    held_out = np.asarray(values, dtype=np.float64)
    if fitted.ndim != 2 or held_out.ndim != 2:
        raise ValueError("train and values must be two-dimensional")
    if fitted.shape[0] == 0 or fitted.shape[1] == 0:
        raise ValueError("training data needs rows and features")
    if held_out.shape[1] != fitted.shape[1]:
        raise ValueError("train and values must have the same features")
    for matrix in (fitted,):
        if not np.all(np.isfinite(matrix)) or np.any(np.abs(matrix) > 1e100):
            raise ValueError("entries must be finite with magnitude at most 1e100")

    # Learn only from training rows. Constant columns have an exact known mean.
    constant = np.all(fitted == fitted[0], axis=0)
    mean = np.mean(fitted, axis=0)
    mean = np.where(constant, fitted[0], mean)
    centered = fitted - mean
    # Rescale before squaring so very small/large nonzero scales survive.
    peak = np.max(np.abs(centered), axis=0)
    safe_peak = np.where(peak == 0, 1.0, peak)
    scale = safe_peak * np.sqrt(np.mean((centered / safe_peak) ** 2, axis=0))
    scale = np.where(constant, 1.0, scale)
    with np.errstate(over="ignore"):
        transformed = (held_out - mean) / scale
    return transformed, mean, scale
