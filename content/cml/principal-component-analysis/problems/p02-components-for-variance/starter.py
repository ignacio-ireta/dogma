from __future__ import annotations

from numpy.typing import ArrayLike


def components_for_variance(X: ArrayLike, threshold: float) -> int:
    """Return the smallest k >= 1 whose first k explained-variance ratios reach ``threshold``.

    Rows of ``X`` are observations and columns are variables. Centre the columns; explained
    variances use the divisor n - 1; each ratio divides by the total variance of all columns.
    A cumulative ratio that equals ``threshold`` in exact arithmetic counts as reaching it: allow a
    rounding margin of 1e-12. Raise ValueError for invalid input: not 2-D, fewer than 2 rows, no
    columns, non-finite entries, a threshold that is not a real number in (0, 1] (booleans
    rejected), or constant data (zero total variance). Do not mutate X.

    Parameters
    ----------
    X : (n, d) real array-like, n >= 2, d >= 1, finite (integers allowed).
    threshold : real number in (0, 1].

    Returns
    -------
    int, the number of leading components needed.
    """
    raise NotImplementedError
