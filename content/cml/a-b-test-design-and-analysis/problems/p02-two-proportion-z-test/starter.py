from __future__ import annotations


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
    raise NotImplementedError
