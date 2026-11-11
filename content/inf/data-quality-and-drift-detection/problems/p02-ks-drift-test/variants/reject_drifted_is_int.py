from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def ks_drift(
    reference: Sequence[float] | NDArray[np.float64],
    current: Sequence[float] | NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Wrong on purpose: the drift flag is returned as an integer.

    Two-sample Kolmogorov-Smirnov drift check: (statistic, p_value, drifted).

    The statistic is the two-sided supremum distance between the two empirical CDFs:

        D = max over x of |F_ref(x) - F_cur(x)|

    Both CDFs are evaluated at every pooled value after all its ties are counted, so tied
    values never inflate or hide the gap.

    The p-value is the limiting Kolmogorov distribution at the effective sample size
    n_e = n * m / (n + m):

        lambda = sqrt(n_e) * D,   p = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lambda^2)

    with p = 1 when lambda = 0. It is an approximation: it is accurate for samples of roughly
    a hundred values or more, and not exact for very small samples.

    ``drifted`` is True when p < alpha.

    Returns
    -------
    (statistic, p_value, drifted) : floats and a bool.
    """
    ref = _finite_1d(reference, "reference")
    cur = _finite_1d(current, "current")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    n, m = ref.size, cur.size
    ref_sorted = np.sort(ref)
    cur_sorted = np.sort(cur)
    pooled = np.concatenate([ref_sorted, cur_sorted])
    # searchsorted(side="right") counts every value <= x, so a whole block of ties is included
    # at once. Evaluating only at the sorted positions would split a tie block.
    f_ref = np.searchsorted(ref_sorted, pooled, side="right") / n
    f_cur = np.searchsorted(cur_sorted, pooled, side="right") / m
    statistic = float(np.max(np.abs(f_ref - f_cur)))

    n_eff = n * m / (n + m)
    p_value = _kolmogorov_sf(math.sqrt(n_eff) * statistic)
    return statistic, p_value, int(p_value < alpha)


def _kolmogorov_sf(lam: float) -> float:
    """P(K > lam) for the limiting Kolmogorov distribution."""
    if lam <= 0.0:
        return 1.0
    if lam < 1.0:
        # The alternating series needs about 1 / lam^2 terms here, so its truncated sum is wrong
        # for small lam (it is off by almost 1 near lam = 1e-6). The equivalent theta series for
        # the CDF converges in a few terms in this range; the p-value is its complement.
        decay = math.pi**2 / (8.0 * lam * lam)
        cdf = (
            math.sqrt(2.0 * math.pi)
            / lam
            * sum(math.exp(-((2 * k - 1) ** 2) * decay) for k in range(1, 50))
        )
        return min(max(1.0 - cdf, 0.0), 1.0)
    total = 0.0
    for k in range(1, 100_001):
        term = math.exp(-2.0 * k * k * lam * lam)
        total += (-1.0) ** (k - 1) * term
        if term < 1e-17:
            break
    return min(max(2.0 * total, 0.0), 1.0)


def _finite_1d(x: object, name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite numbers")
    return values
