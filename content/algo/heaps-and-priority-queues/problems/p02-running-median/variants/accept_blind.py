from __future__ import annotations

import heapq
import math
from collections.abc import Sequence


def running_median(values: Sequence[float]) -> list[float]:
    """Median of every prefix of values, computed with two heaps in O(n log n)."""
    for v in values:
        if not math.isfinite(v):
            raise ValueError("values must be finite")

    low: list[float] = []  # max-heap via negation: lower half
    high: list[float] = []  # min-heap: upper half
    out: list[float] = []
    for v in values:
        x = float(v)
        if not low or x <= -low[0]:
            heapq.heappush(low, -x)
        else:
            heapq.heappush(high, x)

        # rebalance so that len(low) == len(high) or len(low) == len(high) + 1
        if len(low) > len(high) + 1:
            heapq.heappush(high, -heapq.heappop(low))
        elif len(high) > len(low):
            heapq.heappush(low, -heapq.heappop(high))

        if len(low) > len(high):
            out.append(float(-low[0]))
        else:
            a = -low[0]
            b = high[0]
            out.append(float(a / 2.0 + b / 2.0))
    return out
