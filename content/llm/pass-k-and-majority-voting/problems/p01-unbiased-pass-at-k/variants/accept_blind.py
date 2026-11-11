from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    for v in (n, c, k):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError("n, c, k must be int")
    if n < 1 or c < 0 or c > n or k < 1 or k > n:
        raise ValueError("invalid arguments")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    f = n - c
    if k <= 2_000_000:
        s = 0.0
        for i in range(k):
            s += math.log1p(-c / (n - i))
    else:
        s = (
            math.lgamma(f + 1)
            - math.lgamma(f - k + 1)
            - math.lgamma(n + 1)
            + math.lgamma(n - k + 1)
        )
    r = -math.expm1(s)
    return min(1.0, max(0.0, r))
