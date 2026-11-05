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
    """Return the percentile bootstrap interval ``(lo, hi)`` of ``statistic`` on ``x``.

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
    """
    raise NotImplementedError
