from __future__ import annotations

import math
from numbers import Integral

from scipy.stats import norm


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer count")
    return int(value)


def two_proportion_z_test(x_a: int, n_a: int, x_b: int, n_b: int) -> tuple[float, float]:
    """Return ``(z, p_two_sided)`` for the difference of two conversion rates, B minus A.

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
    """
    xa = _count(x_a, "x_a")
    na = _count(n_a, "n_a")
    xb = _count(x_b, "x_b")
    nb = _count(n_b, "n_b")
    if na < 1 or nb < 1:
        raise ValueError("n_a and n_b must be at least 1")
    if not (0 <= xa <= na) or not (0 <= xb <= nb):
        raise ValueError("each count must lie in [0, n]")

    p_a = xa / na
    p_b = xb / nb
    p_pool = (xa + xb) / (na + nb)
    if p_pool <= 0.0 or p_pool >= 1.0:
        raise ValueError("pooled rate is 0 or 1: the z statistic is undefined")

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / na + 1.0 / nb))
    z = (p_b - p_a) / se
    p = float(2.0 * norm.sf(abs(z)))
    return (float(z), p)
