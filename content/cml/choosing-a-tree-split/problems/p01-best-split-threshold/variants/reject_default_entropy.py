from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_TIE_TOLERANCE = 1e-12  # a later candidate must beat the incumbent by more than this


def _impurity(
    counts: NDArray[np.float64], sizes: NDArray[np.float64], criterion: str
) -> NDArray[np.float64]:
    """Impurity of each node. counts has one row per node; sizes[i] is row i's total count."""
    p = counts / sizes[:, None]
    if criterion == "gini":
        # G = 1 - sum_k p_k^2
        return 1.0 - np.sum(p * p, axis=1)
    # H = -sum_k p_k log2 p_k in bits. Zero proportions contribute 0 (0 log 0 = 0), so the
    # logarithm is taken only where p > 0, with a placeholder elsewhere to avoid warnings.
    safe = np.where(p > 0.0, p, 1.0)
    return -np.sum(np.where(p > 0.0, p * np.log2(safe), 0.0), axis=1)


def best_split(
    x: ArrayLike, y: ArrayLike, *, criterion: str = "entropy"
) -> tuple[float | None, float]:
    """Wrong on purpose: the default criterion is entropy instead of gini.

    Sort once. Every candidate threshold then splits the sorted order into a prefix (left) and a
    suffix (right), so the class counts on each side come from prefix sums, not from a rescan.
    That makes the search O(n log n) overall. Candidates exist only where the sorted value
    strictly increases, so duplicate values never produce a threshold between equal values.
    """
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'")
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers")
    labels = np.asarray(y, dtype=object)
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("y must be one-dimensional, with one label per value of x")

    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    xs = values[order]

    # One-hot class indicators in sorted order, then prefix sums: prefix[i] holds the class
    # counts of the i + 1 smallest values.
    # Dictionary equality preserves distinct scalar labels such as 1 and "1", without
    # requiring unlike scalar types to be sortable or coercing them to one string dtype.
    classes: dict[object, int] = {}
    codes = np.array([classes.setdefault(label, len(classes)) for label in labels])
    one_hot = np.zeros((n, int(codes.max()) + 1))
    one_hot[np.arange(n), codes[order]] = 1.0
    prefix = np.cumsum(one_hot, axis=0)
    totals = prefix[-1]

    # Candidate after sorted position i exists only where the next value is strictly larger.
    cut = np.flatnonzero(xs[1:] > xs[:-1])
    if cut.size == 0:
        return None, 0.0

    n_left = (cut + 1).astype(np.float64)
    n_right = n - n_left
    left_counts = prefix[cut]
    right_counts = totals - left_counts

    parent = _impurity(totals[None, :], np.array([float(n)]), criterion)[0]
    children = (n_left / n) * _impurity(left_counts, n_left, criterion) + (n_right / n) * _impurity(
        right_counts, n_right, criterion
    )
    decrease = parent - children
    # Halve first so finite same-sign endpoints cannot overflow on addition.
    thresholds = xs[cut] / 2.0 + xs[cut + 1] / 2.0

    # Scan in increasing threshold order. A later candidate replaces the incumbent only when it
    # is better by more than the tolerance, so exact ties keep the smallest threshold.
    best = 0
    for j in range(1, decrease.shape[0]):
        if decrease[j] > decrease[best] + _TIE_TOLERANCE:
            best = j
    return float(thresholds[best]), float(decrease[best])
