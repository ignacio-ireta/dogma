from __future__ import annotations

import math
from collections.abc import Iterable


def welford(xs: Iterable[float]) -> tuple[int, float, float]:
    """Count, mean and sample variance of ``xs`` in one pass with Welford's update.

    Returns (n, mean, variance) with variance = M2 / (n - 1), nan when n == 1.
    Raises ValueError when xs is empty.
    """
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in xs:
        value = float(x)
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
    if n == 0:
        raise ValueError("welford needs at least one value")
    variance = m2 / (n - 1) if n > 1 else math.nan
    return n, mean, variance
