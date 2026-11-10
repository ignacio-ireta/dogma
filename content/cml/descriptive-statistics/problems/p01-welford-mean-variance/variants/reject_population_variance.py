from __future__ import annotations

import math
from collections.abc import Iterable


def welford(xs: Iterable[float]) -> tuple[int, float, float]:
    """Wrong on purpose: divides M2 by n, not n - 1. Count, mean and sample variance of ``xs`` in one pass with Welford's update.

    Keeps three running numbers: the count n, the mean, and M2 = sum of (x - mean)^2. Each new
    value x updates the mean first and then M2, so no sum of squares of the raw values is formed.
    That avoids the cancellation in E[x^2] - E[x]^2 when the values are large relative to their
    spread.

    Returns
    -------
    (n, mean, variance) : n is an int; variance is M2 / (n - 1), or nan when n == 1.
    """  # noqa: E501
    count = 0
    mean = 0.0
    m2 = 0.0
    for value in xs:
        x = float(value)
        count += 1
        # Deviation from the mean before this value is included.
        delta = x - mean
        mean += delta / count
        # The updated mean gives the second factor. Together they add (x - old)(x - new) to M2.
        m2 += delta * (x - mean)
    if count == 0:
        raise ValueError("welford needs at least one value")
    variance = m2 / count if count > 1 else math.nan
    return count, mean, variance
