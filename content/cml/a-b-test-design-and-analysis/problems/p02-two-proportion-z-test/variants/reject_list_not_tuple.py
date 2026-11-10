from __future__ import annotations

import math
import numbers

from scipy import stats


def _count(value: object, name: str) -> int:
    """Accept Python and NumPy integers; reject floats and booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer count")
    return int(value)


def two_proportion_z_test(x_a: int, n_a: int, x_b: int, n_b: int) -> tuple[float, float]:
    """Wrong on purpose: return ``(z, p_two_sided)`` for the difference of two conversion rates, B minus A.

    Parameters
    ----------
    x_a, x_b : integer conversions in arm A (control) and arm B (treatment).
    n_a, n_b : integer users in each arm, at least 1.

    Returns
    -------
    (z, p) with ``p_pool = (x_a + x_b) / (n_a + n_b)``,
    ``z = (x_b / n_b - x_a / n_a) / sqrt(p_pool (1 - p_pool) (1/n_a + 1/n_b))`` and
    ``p = 2 * P(Z >= |z|)`` for a standard normal Z.

    Raises
    ------
    ValueError
        If a count is not an integer, ``n_a < 1`` or ``n_b < 1``, a count is outside ``[0, n]``,
        or the pooled rate is 0 or 1.
    """  # noqa: E501
    x_a, n_a = _count(x_a, "x_a"), _count(n_a, "n_a")
    x_b, n_b = _count(x_b, "x_b"), _count(n_b, "n_b")
    if n_a < 1 or n_b < 1:
        raise ValueError("each arm needs at least one user")
    if not (0 <= x_a <= n_a and 0 <= x_b <= n_b):
        raise ValueError("conversions must lie between 0 and the number of users in the arm")

    # Under H0 both arms share one rate, so the standard error uses the pooled rate.
    pooled = (x_a + x_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        raise ValueError("pooled rate is 0 or 1: the z statistic is undefined")
    p_a = x_a / n_a
    p_b = x_b / n_b
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    z_stat = (p_b - p_a) / standard_error
    # The survival function keeps tiny tail probabilities that 1 - cdf would round to zero.
    p_value = 2.0 * float(stats.norm.sf(abs(z_stat)))
    return [float(z_stat), p_value]
