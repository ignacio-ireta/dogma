from __future__ import annotations

from numpy.typing import ArrayLike


def welch_t_test(a: ArrayLike, b: ArrayLike) -> tuple[float, float, float]:
    """Return ``(t, df, p_two_sided)`` for Welch's two-sample t-test of ``a`` against ``b``.

    Parameters
    ----------
    a, b : 1-D array-likes of n >= 2 finite real numbers each (ints allowed).

    Returns
    -------
    (t, df, p) where ``t = (mean(a) - mean(b)) / sqrt(va + vb)``, ``va = var(a, ddof=1) / n_a``,
    ``vb = var(b, ddof=1) / n_b``, ``df = (va + vb)**2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))``
    (Welch-Satterthwaite), and ``p = 2 * P(T_df >= |t|)``.

    Raises
    ------
    ValueError
        If an input is not 1-D, has fewer than two values, contains NaN or infinity, or if both
        samples are constant (the standard error is zero).
    """
    raise NotImplementedError
