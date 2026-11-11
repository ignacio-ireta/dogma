from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def correlation_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Return the (d, d) Pearson correlation matrix of the (n, d) data matrix ``X``.

    Entry [j, k] is C[j, k] / sqrt(C[j, j] * C[k, k]). Constant columns get NaN, by convention.
    """
    x = np.asarray(X, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array (observations, variables), got {x.ndim}-D")
    n, d = x.shape
    if n == 0 or d == 0:
        raise ValueError("need at least one observation and one variable")
    if n < 2:
        raise ValueError("need at least two observations to measure correlation")
    if not np.isfinite(x).all():
        raise ValueError("observations must be finite (no NaN or inf)")

    # A column is constant when all its entries are equal. Test that exactly, on the raw data:
    # the mean of repeated values can differ from the value (mean([0.1] * 3) != 0.1), so a
    # computed variance would call a constant column non-constant.
    constant = x.max(axis=0) == x.min(axis=0)

    # Correlation is unchanged when each column is multiplied by a positive number. Dividing by
    # the largest magnitude first keeps every square in range: raw squares overflow near 1e154
    # and underflow near 1e-162.
    peak = np.max(np.abs(x), axis=0)
    safe_peak = np.where(peak > 0, peak, 1.0)
    z = x / safe_peak

    centred = z - z.mean(axis=0)
    # Scale each column to unit length first, so the product is the correlation directly.
    norms = np.sqrt((centred * centred).sum(axis=0))
    unit = centred / np.where(constant, 1.0, norms)
    corr = np.clip(unit.T @ unit, -1.0, 1.0)

    corr[constant, :] = np.nan
    corr[:, constant] = np.nan
    return corr
