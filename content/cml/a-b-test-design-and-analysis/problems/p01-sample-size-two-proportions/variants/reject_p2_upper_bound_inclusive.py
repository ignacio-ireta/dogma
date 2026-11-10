from __future__ import annotations

import math

from scipy import stats


def sample_size_two_proportions(
    p_baseline: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Wrong on purpose: return the users needed in each arm to detect an absolute change ``mde`` in a rate.

    Parameters
    ----------
    p_baseline : control rate, strictly between 0 and 1.
    mde : absolute effect, nonzero; the treatment rate ``p_baseline + mde`` must lie in (0, 1).
    alpha : two-sided significance level, strictly between 0 and 1.
    power : target power, strictly between 0 and 1.

    Returns
    -------
    int: ``ceil(n)``, where ``n = (z_a sqrt(2 pbar (1 - pbar)) + z_b sqrt(p1 (1 - p1)
    + p2 (1 - p2)))**2 / (p2 - p1)**2``, ``p1 = p_baseline``, ``p2 = p1 + mde``,
    ``pbar = (p1 + p2) / 2``, ``z_a = Phi^-1(1 - alpha / 2)`` and ``z_b = Phi^-1(power)``.

    Raises
    ------
    ValueError
        If ``p_baseline`` or ``p1 + mde`` is not strictly between 0 and 1, if ``mde == 0``, or if
        ``alpha`` or ``power`` is not strictly between 0 and 1.
    """  # noqa: E501
    # Written as "not (0 < x < 1)" so that NaN is rejected as well.
    if not 0.0 < p_baseline < 1.0:
        raise ValueError("p_baseline must be strictly between 0 and 1")
    if mde == 0:
        raise ValueError("mde must be nonzero: a zero effect has no sample size to plan for")
    p1 = float(p_baseline)
    p2 = p1 + float(mde)
    if not 0.0 < p2 <= 1.0:
        raise ValueError("p_baseline + mde must stay strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be strictly between 0 and 1")

    # Under H0 both arms share one rate, so the pooled rate gives the variance of the difference.
    pbar = (p1 + p2) / 2.0
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))  # two-sided critical value
    # Under H1 the two arms have their own rates, so their variances are added unpooled.
    z_beta = float(stats.norm.ppf(power))  # power = 1 - beta
    numerator = (
        z_alpha * math.sqrt(2.0 * pbar * (1.0 - pbar))
        + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    n_per_arm = numerator / float(mde) ** 2
    return math.ceil(n_per_arm)
