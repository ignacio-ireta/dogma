from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Return the softmax of logits along axis, computed stably for every finite logit.

    Parameters
    ----------
    logits : array-like of finite real numbers with at least one dimension.
    axis : the axis to normalise over. Every other axis is an independent distribution.

    Returns
    -------
    float64 array with the shape of logits. Along axis, each slice is non-negative and sums to 1.

    Raises
    ------
    ValueError
        For a scalar, NaN or infinite entries, an empty axis, or an out-of-range axis.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("softmax needs an array with at least one dimension")
    if not np.all(np.isfinite(x)):
        raise ValueError("softmax needs finite logits")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f"axis {axis} is out of range for {x.ndim} dimensions")
    if x.shape[axis] == 0:
        raise ValueError("softmax needs a non-empty axis")
    with np.errstate(over="ignore"):
        # z - max is <= 0; it can only overflow to -inf, which exp maps to 0 correctly.
        shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
