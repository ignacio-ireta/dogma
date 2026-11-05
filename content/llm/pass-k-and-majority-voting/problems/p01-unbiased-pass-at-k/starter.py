from __future__ import annotations


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased pass@k: 1 - C(n - c, k) / C(n, k).

    n >= 1 samples were drawn, c (0 <= c <= n) of them pass, and 1 <= k <= n. Return 0.0 when
    c == 0 and exactly 1.0 when n - c < k. Keep relative accuracy near 1e-9 even when the answer is
    tiny, and never build large binomial coefficients as floats. Raise TypeError for non-int
    arguments (bool is rejected), and ValueError for n < 1, c outside [0, n], or k outside [1, n].
    """
    raise NotImplementedError
