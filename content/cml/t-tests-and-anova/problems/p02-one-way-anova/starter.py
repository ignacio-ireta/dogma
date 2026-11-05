from __future__ import annotations

from numpy.typing import ArrayLike


def one_way_anova(*groups: ArrayLike) -> tuple[float, float]:
    """Return ``(F, p)`` for the one-way ANOVA of the given groups.

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
    raise NotImplementedError
