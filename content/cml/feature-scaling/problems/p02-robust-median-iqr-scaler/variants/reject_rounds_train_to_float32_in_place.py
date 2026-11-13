from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# sklearn treats a scale below 10 machine epsilons as zero and replaces it by one.
_TINY_SCALE = 10 * np.finfo(np.float64).eps


def _linear_quantile(sorted_column: NDArray[np.float64], q: float) -> float:
    """Quantile by linear interpolation between order statistics (NumPy's default method).

    The position is h = q * (n - 1). The value interpolates between the two neighbouring order
    statistics, and the upper index is capped at n - 1.
    """
    n = sorted_column.shape[0]
    position = q * (n - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, n - 1)
    fraction = position - lower
    left, right = sorted_column[lower], sorted_column[upper]
    # Opposite signs can overflow right-left although the interpolated value is finite.
    if left < 0.0 < right:
        return float((1.0 - fraction) * left + fraction * right)
    return float(left + fraction * (right - left))


def robust_standardize(
    train: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return values centered by the training median and scaled by the training IQR.

    train is (n, d), n >= 1, d >= 1; values is (m, d), m >= 0. Entries must be finite.
    The scale is the 25th-to-75th percentile range with linear interpolation; a column whose
    scale is below 10 machine epsilons gets scale 1.0. Never mutate inputs.
    Returns float64 arrays of shapes (m, d), (d,), (d,): transformed, center, scale.
    """
    fitted = np.asarray(train, dtype=np.float64)
    held_out = np.asarray(values, dtype=np.float64)
    if fitted.ndim != 2 or held_out.ndim != 2:
        raise ValueError("train and values must be two-dimensional")
    if fitted.shape[0] == 0 or fitted.shape[1] == 0:
        raise ValueError("training data needs rows and features")
    if held_out.shape[1] != fitted.shape[1]:
        raise ValueError("train and values must have the same features")
    if not np.all(np.isfinite(fitted)) or not np.all(np.isfinite(held_out)):
        raise ValueError("entries must be finite")

    # Learn only from the training rows. Sorting copies each column, so the input is untouched.
    center = np.empty(fitted.shape[1], dtype=np.float64)
    scale = np.empty(fitted.shape[1], dtype=np.float64)
    for j in range(fitted.shape[1]):
        column = np.sort(fitted[:, j])
        center[j] = _linear_quantile(column, 0.5)  # the median
        iqr = _linear_quantile(column, 0.75) - _linear_quantile(column, 0.25)
        scale[j] = 1.0 if iqr < _TINY_SCALE else iqr

    # Apply the training center and scale to every row, including rows far outside the IQR.
    transformed = (held_out - center) / scale
    return transformed, center, scale


_correct = robust_standardize


def _change_in_place(v):
    if v.dtype.kind == "f":
        v[...] = v.astype("float32")


def robust_standardize(*args, **kwargs):
    """Wrong on purpose: the result is right, but this rounds to float32 the values of `train` in place."""  # noqa: E501
    result = _correct(*args, **kwargs)
    target = args[0] if len(args) > 0 else kwargs.get("train")
    if type(target).__name__ in ("ndarray",):
        _change_in_place(target)
    return result


robust_standardize.__wrapped__ = _correct
