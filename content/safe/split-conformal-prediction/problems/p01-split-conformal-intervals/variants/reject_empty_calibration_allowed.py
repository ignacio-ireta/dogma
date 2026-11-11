from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _real_vector(values: Any, name: str, *, allow_empty: bool) -> NDArray[np.float64]:
    """Validate a real 1-D array before any arithmetic touches it."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    array = raw.astype(np.float64)
    if array.ndim != 1 or (array.size == 0 and not allow_empty):
        raise ValueError(f"{name} must be a 1-D array, nonempty where required")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rank(n: int, alpha: float) -> int:
    """Return k = ceil((n + 1)(1 - alpha)) with alpha read as the decimal it was written as.

    repr(0.7) is "0.7", so the rank uses exactly 7/10. The exact binary value of the float is
    not 7/10, and float products such as 10 * 0.3 can land a hair above an integer, so both
    routes can move the rank by one. Fraction arithmetic keeps the decimal value exact.
    """
    exact_alpha = Fraction(repr(float(alpha)))
    return math.ceil((n + 1) * (1 - exact_alpha))


def split_conformal_interval(
    cal_pred: NDArray[np.float64],
    cal_true: NDArray[np.float64],
    test_pred: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: empty calibration set accepted, giving an infinite interval. Return (lower, upper) split conformal bounds for each test prediction."""  # noqa: E501
    # bool is an int subclass, so reject it before the numeric range test.
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a float in (0, 1)")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a float in (0, 1)")
    pred = _real_vector(cal_pred, "cal_pred", allow_empty=True)
    true = _real_vector(cal_true, "cal_true", allow_empty=True)
    if pred.shape != true.shape:
        raise ValueError("cal_pred and cal_true must have the same length")
    test = _real_vector(test_pred, "test_pred", allow_empty=True)

    n = pred.shape[0]
    # Calibration scores: absolute residuals, sorted so the rank indexes them directly.
    scores = np.sort(np.abs(true - pred))
    k = _rank(n, float(alpha))
    # Rank above n means the calibration set cannot certify this level: no finite threshold.
    q = math.inf if k > n else float(scores[k - 1])
    return test - q, test + q
