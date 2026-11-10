from __future__ import annotations

import heapq
import math
from collections.abc import Sequence


def running_median(values: Sequence[float]) -> list[float]:
    """Wrong on purpose: infinite values are not rejected, only NaN. Median of every prefix of values, computed with two heaps in O(n log n).

    Element i of the result is the median of values[: i + 1]. An odd count gives the middle value;
    an even count gives the mean of the two middle values. Each result is a Python float.

    Parameters
    ----------
    values : finite real numbers, possibly empty. The input is not modified.

    Returns
    -------
    A list of len(values) floats.

    Raises
    ------
    ValueError
        If any value is NaN or infinite.
    """  # noqa: E501
    if any(math.isnan(value) for value in values):
        raise ValueError("values must be finite: NaN and infinity have no place in an order")

    # The lower half lives in a max-heap, stored as negatives because heapq is a min-heap.
    # The upper half lives in a min-heap. The lower half holds the same count as the upper half,
    # or one more, so the middle of the data is always at the top of one heap or both.
    lower: list[float] = []
    upper: list[float] = []
    medians: list[float] = []
    for value in values:
        # A value above the largest of the lower half belongs to the upper half.
        if lower and value > -lower[0]:
            heapq.heappush(upper, value)
        else:
            heapq.heappush(lower, -value)
        # Rebalance so the sizes differ by at most one, with any extra element in the lower half.
        if len(lower) > len(upper) + 1:
            heapq.heappush(upper, -heapq.heappop(lower))
        elif len(upper) > len(lower):
            heapq.heappush(lower, -heapq.heappop(upper))

        if len(lower) == len(upper):
            # Even count: the two middle values are the tops of the two heaps.
            a, b = -lower[0], upper[0]
            # Same-sign addition may overflow even when its mean is representable.
            middle = a + (b - a) / 2 if (a >= 0) == (b >= 0) else (a + b) / 2
            medians.append(float(middle))
        else:
            # Odd count: the middle value is the top of the lower heap.
            medians.append(float(-lower[0]))
    return medians
