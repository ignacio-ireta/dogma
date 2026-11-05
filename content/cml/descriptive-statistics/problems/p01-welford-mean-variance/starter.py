from __future__ import annotations

from collections.abc import Iterable


def welford(xs: Iterable[float]) -> tuple[int, float, float]:
    """Count, mean and sample variance of ``xs`` in one pass with Welford's update.

    ``xs`` is any iterable of real numbers, including a generator that can be consumed only once.
    Use a single pass with O(1) memory: keep a count, a running mean and M2 = sum of
    (x - mean)^2, updated per value.

    Returns
    -------
    (n, mean, variance) : n is an int; variance is the sample variance, M2 / (n - 1), and it is
        nan when n == 1.

    Raises
    ------
    ValueError
        If ``xs`` is empty.
    """
    raise NotImplementedError
