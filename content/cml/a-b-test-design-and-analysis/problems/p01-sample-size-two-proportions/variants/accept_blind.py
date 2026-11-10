from __future__ import annotations

import math

from scipy.stats import norm


def sample_size_two_proportions(
    p_baseline: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Return the users needed in each arm to detect an absolute change ``mde`` in a rate.

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
    """
    p1 = float(p_baseline)
    d = float(mde)
    a = float(alpha)
    b = float(power)
    if not (0.0 < p1 < 1.0):
        raise ValueError("p_baseline must be strictly between 0 and 1")
    if d == 0.0 or math.isnan(d):
        raise ValueError("mde must be nonzero")
    p2 = p1 + d
    if not (0.0 < p2 < 1.0):
        raise ValueError("p_baseline + mde must stay strictly between 0 and 1")
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not (0.0 < b < 1.0):
        raise ValueError("power must be strictly between 0 and 1")

    pbar = (p1 + p2) / 2.0
    z_a = float(norm.ppf(1.0 - a / 2.0))
    z_b = float(norm.ppf(b))
    numerator = (
        z_a * math.sqrt(2.0 * pbar * (1.0 - pbar))
        + z_b * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    n = numerator / (p2 - p1) ** 2
    return int(math.ceil(n))  # noqa: RUF046
