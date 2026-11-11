from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _probability_matrix(values: Any, name: str, *, allow_empty: bool) -> NDArray[np.float64]:
    """Validate a (rows, K) probability matrix before any arithmetic touches it."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    array = raw.astype(np.float64)
    if array.ndim != 2 or array.shape[1] < 2 or (array.shape[0] == 0 and not allow_empty):
        raise ValueError(f"{name} must have shape (rows, K) with K >= 2")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if array.shape[0] and np.any(np.abs(array.sum(axis=1) - 1.0) > 1e-9):
        raise ValueError(f"every row of {name} must sum to 1")
    return array


def _labels(values: Any, n: int, classes: int) -> NDArray[np.intp]:
    """Validate integer labels: bool and float dtypes are rejected before any indexing."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise ValueError("cal_labels must be an integer array")
    if raw.shape != (n,):
        raise ValueError("cal_labels must have one entry per calibration row")
    if np.any(raw < 0) or np.any(raw >= classes):
        raise ValueError("cal_labels must lie in [0, K)")
    return raw.astype(np.intp)


def _rank(n: int, alpha: float) -> int:
    """Return k = ceil((n + 1)(1 - alpha)) with alpha read as the decimal it was written as.

    repr(0.7) is "0.7", so the rank uses exactly 7/10. The exact binary value of the float is
    not 7/10, and float products such as 10 * 0.3 can land a hair above an integer, so both
    routes can move the rank by one. Fraction arithmetic keeps the decimal value exact.
    """
    exact_alpha = Fraction(repr(float(alpha)))
    return math.ceil((n + 1) * (1 - exact_alpha))


def prediction_sets(
    cal_probs: Any,
    cal_labels: Any,
    test_probs: Any,
    alpha: Any,
) -> NDArray[np.bool_]:
    """Wrong on purpose: accepts alpha equal to zero. Return a boolean (m, K) mask of classes in each conformal prediction set."""  # noqa: E501
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a float in (0, 1)")
    if not math.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be a float in (0, 1)")
    cal = _probability_matrix(cal_probs, "cal_probs", allow_empty=False)
    n, classes = cal.shape
    test = _probability_matrix(test_probs, "test_probs", allow_empty=True)
    if test.shape[1] != classes:
        raise ValueError("cal_probs and test_probs must have the same number of classes")
    labels = _labels(cal_labels, n, classes)

    # Score each calibration point by one minus the probability of its true class.
    scores = np.sort(1.0 - cal[np.arange(n), labels])
    k = _rank(n, float(alpha))
    # The threshold is +inf when the rank exceeds n: every class is then in every set.
    q = math.inf if k > n else float(scores[k - 1])
    # Score each test class the same way and keep the classes at or under the threshold.
    return np.asarray((1.0 - test) <= q, dtype=bool)
