from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# A cumulative ratio within this margin below the threshold counts as reaching it.
# Without the margin, a tie in exact arithmetic (such as 0.75 = 3/4) can be lost to rounding.
# a tie in exact arithmetic (such as 0.75 = 3/4) can be lost to a few units of rounding.
_ROUNDING_MARGIN = 1e-12


def components_for_variance(X: ArrayLike, threshold: float) -> int:
    """Wrong on purpose: return the smallest k >= 1 whose first k explained-variance ratios reach ``threshold``.

    The ratios come from the same definitions as PCA: centred columns, sample variances with divisor
    n - 1, and each ratio divided by the total variance of all columns.
    """  # noqa: E501
    # np.array copies by default, so the caller's matrix is never modified.
    x = np.array(X, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"X must be 2-D (observations, variables), got {x.ndim}-D")
    n, d = x.shape
    if n < 2 or d < 1:
        raise ValueError(
            "X needs at least two rows and one column: a sample variance needs two rows"
        )
    if not np.isfinite(x).all():
        raise ValueError("X must contain only finite numbers (no NaN or inf)")
    # Accept real numbers only; booleans are excluded explicitly because bool is a subclass of int.
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("threshold must be a real number in (0, 1]")
    t = float(threshold)
    # Written as "not (... <= ...)" so that NaN is rejected too.
    if not 0.0 < t <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")

    # Centre, then take the singular values. The variance of component i is s_i**2 / (n - 1).
    centred = x - x.mean(axis=0)
    _, s, _ = np.linalg.svd(centred, full_matrices=False)
    explained_variance = s**2 / n

    # Each ratio divides by the total variance of all columns, the same denominator as PCA.
    total_variance = float(np.var(x, axis=0, ddof=1).sum())
    if total_variance == 0.0:
        raise ValueError("every column is constant: the total variance is zero, so no ratio exists")
    ratio = explained_variance / total_variance

    # The cumulative sum is non-decreasing and ends at 1, so some k always reaches the threshold.
    cumulative = np.cumsum(ratio)
    reached = np.flatnonzero(cumulative >= t - _ROUNDING_MARGIN)
    # Convert the zero-based index of the first reaching component into a count.
    return int(reached[0]) + 1
