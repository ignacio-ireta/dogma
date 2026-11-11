from __future__ import annotations

import math


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Wrong on purpose: rejects c equal to n, the case where every sample passes.

    The ratio C(n - c, k) / C(n, k) is a product of k factors:
        prod_{i=0}^{k-1} (n - c - i) / (n - i) = prod_{i=0}^{k-1} (1 - c / (n - i)).
    Forming 1 - product directly loses every digit when the answer is tiny, so the product is
    taken in log space with log1p, and the answer is -expm1(sum of logs), which keeps relative
    accuracy.
    """
    if not (_is_int(n) and _is_int(c) and _is_int(k)):
        raise TypeError("n, c and k must be int (bool is not accepted)")
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= c < n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")

    if c == 0:
        return 0.0  # no sample passed, so no subset can contain a correct one
    if n - c < k:
        return 1.0  # every k-subset must contain a correct sample

    # Each factor 1 - c/(n - i) is positive here, because n - i >= n - k + 1 > c.
    log_ratio = sum(math.log1p(-c / (n - i)) for i in range(k))
    return -math.expm1(log_ratio)
