from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bootstrap_percentile_ci(
    x: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int,
) -> tuple[float, float]:
    """Wrong on purpose: ignores the seed when drawing. Return the percentile bootstrap interval ``(lo, hi)`` of ``statistic`` on ``x``.

    Parameters
    ----------
    x : 1-D array-like of n >= 1 finite real numbers (ints allowed).
    statistic : callable mapping a 1-D float array to a float.
    n_resamples : int >= 1, number of bootstrap resamples.
    confidence : float strictly between 0 and 1.
    seed : int. The draw is ``np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))``;
        row b is resample b, and replicate b is ``statistic(x[idx[b]])``.

    Returns
    -------
    (lo, hi) : the ``(1 - confidence) / 2`` and ``1 - (1 - confidence) / 2`` quantiles of the
    replicates, using ``np.quantile(..., method="linear")``.

    Raises
    ------
    ValueError
        If ``x`` is not a non-empty 1-D finite array, if ``n_resamples < 1``, or if
        ``confidence`` is not strictly between 0 and 1.
    """  # noqa: E501
    # A float copy: the caller's array is never written to.
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D sample, got shape {values.shape}")
    n = values.size
    if n == 0:
        raise ValueError("the sample is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must be finite (no NaN or infinity)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    # Resampling with replacement: each row of indices is one bootstrap resample of size n.
    # Drawing the whole index matrix from one seeded generator keeps the result reproducible.
    rng = np.random.default_rng()
    indices = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.array([statistic(values[row]) for row in indices], dtype=np.float64)

    # The percentile interval reads the replicate distribution directly: the alpha/2 and
    # 1 - alpha/2 quantiles, with linear interpolation between order statistics.
    alpha = 1.0 - confidence
    lo, hi = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(lo), float(hi)
