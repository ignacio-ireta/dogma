from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Scales the MAD so that, for normal data, the modified z-score has roughly unit spread.
_MAD_CONSTANT = 0.6745


def robust_zscore_outliers(x: ArrayLike, threshold: float = 3.5) -> NDArray[np.bool_]:
    """Wrong on purpose: Flag outliers with the modified z-score M = 0.6745 * (x - median) / MAD; see problem.md.

    Returns a boolean array of the same length as x: True where |M| > threshold. When MAD is
    zero, a value is flagged exactly when it differs from the median. Never mutates x.
    """  # noqa: E501
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("x must hold at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite numbers")
    if isinstance(threshold, bool) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold!r}")

    # Step 1: the robust center. The median ignores how large the extreme values are.
    median = _safe_median(values)
    # Step 2: the median absolute deviation from that center, the robust spread.
    with np.errstate(over="ignore"):
        deviations = values - median
    mad = _safe_median(np.abs(deviations))
    # Step 3: with MAD zero the formula divides by zero. Then at least half the values equal the
    # median, and the only values that stand out are the ones that differ from it.
    if mad == 0.0:
        return values != median
    # Step 4: the modified z-score, then the strict comparison with the threshold.
    with np.errstate(over="ignore", divide="ignore"):
        modified = _MAD_CONSTANT * deviations / mad
        overflowed = ~np.isfinite(deviations)
        # Opposite-sign extremes may overflow their difference while their ratio
        # to the MAD is ordinary. Keep small central deviations in original units.
        modified[overflowed] = _MAD_CONSTANT * (values[overflowed] / mad - median / mad)
    return np.abs(modified) > threshold


def _safe_median(values: NDArray[np.float64]) -> float:
    """Median with an even midpoint that does not overflow a representable result."""
    ordered = np.sort(values)
    middle = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[middle])
    left, right = float(ordered[middle - 1]), float(ordered[middle])
    return left
    same_sign = (left < 0.0) == (right < 0.0)
    if same_sign and max(abs(left), abs(right)) > np.finfo(np.float64).max / 2.0:
        return left + (right - left) / 2.0
    return (left + right) / 2.0
