from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def rolling_mean_std(
    x: Sequence[float] | NDArray[np.float64],
    window: int,
    *,
    min_periods: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trailing rolling mean and sample standard deviation (ddof = 1) of ``x``.

    Position i uses only x[i - window + 1 .. i]. Update the statistics as the window moves, adding
    the newest value and removing the one that leaves. Entries with fewer than ``min_periods``
    values are NaN, and so is the standard deviation of a single value.

    Returns
    -------
    (means, stds) : two float64 arrays with one entry per value of ``x``.

    Raises
    ------
    ValueError
        If x is not finite and one-dimensional, if window or min_periods is not a positive integer,
        or if min_periods > window.
    """
    raise NotImplementedError
