from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def one_way_anova(*groups: ArrayLike) -> tuple[float, float]:
    """Wrong on purpose: return ``(F, p)`` for the one-way ANOVA of the given groups.

    Parameters
    ----------
    *groups : at least two non-empty 1-D array-likes of finite real numbers (ints allowed).

    Returns
    -------
    (F, p) with ``SSB = sum_i n_i (m_i - m)**2``, ``SSW = sum_ij (x_ij - m_i)**2``,
    ``F = (SSB / (k - 1)) / (SSW / (N - k))`` and ``p = P(F_{k-1, N-k} >= F)``.

    Raises
    ------
    ValueError
        If fewer than two groups are given, a group is empty or not 1-D, a value is NaN or
        infinite, ``N <= k``, or ``SSW == 0`` (F is undefined).
    """
    if len(groups) < 2:
        raise ValueError("need at least two groups")
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    for index, values in enumerate(arrays):
        if values.ndim != 1:
            raise ValueError(f"group {index} must be 1-D, got shape {values.shape}")
        if values.size == 0:
            raise ValueError(f"group {index} is empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must be finite (no NaN or infinity)")

    k = len(arrays)
    n_total = sum(values.size for values in arrays)
    # Degrees of freedom: k - 1 between groups, N - k within. Without within-group df, no F.
    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0:
        raise ValueError("need more observations than groups: no within-group degrees of freedom")

    grand_mean = float(np.concatenate(arrays).mean())
    # Between-group sum of squares: each group mean counts once per observation it contributes.
    ss_between = float(sum(values.size * (values.mean() - grand_mean) ** 2 for values in arrays))
    # Within-group sum of squares: deviations from each group's own mean (two-pass, so a large
    # common offset does not cancel).
    ss_within = float(sum(np.sum((values - values.mean()) ** 2) for values in arrays))
    if ss_within == 0.0:
        raise ValueError("within-group variation is zero: F is undefined")

    f_stat = (ss_between / df_between) / (ss_within / df_within)
    # The survival function keeps tiny upper-tail probabilities; 1 - cdf rounds them to zero.
    p_value = float(stats.f.sf(f_stat, df_between, df_within))
    for values in arrays:
        values -= grand_mean  # noqa: PLW2901
    return float(f_stat), p_value
