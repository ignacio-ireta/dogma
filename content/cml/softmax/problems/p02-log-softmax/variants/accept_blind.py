from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def log_softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Return log(softmax(logits)) along axis, computed stably via the log-sum-exp.

    Parameters
    ----------
    logits : array-like of finite real numbers with at least one dimension.
    axis : the axis to normalise over. Every other axis is an independent distribution.

    Returns
    -------
    float64 array with the shape of logits. Along axis, exp of the result sums to 1, and the
    representable log-probabilities are finite; an out-of-range negative tail is -inf.

    Raises
    ------
    ValueError
        For a scalar, NaN or infinite entries, an empty axis, or an out-of-range axis.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("log_softmax needs an array with at least one dimension")
    if not np.all(np.isfinite(x)):
        raise ValueError("log_softmax needs finite logits")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f"axis {axis} is out of range for {x.ndim} dimensions")
    if x.shape[axis] == 0:
        raise ValueError("log_softmax needs a non-empty axis")
    with np.errstate(over="ignore"):
        # Entries that are out of float64 range below the max become -inf (not a warning).
        shifted = x - np.max(x, axis=axis, keepdims=True)
    # The sum includes the max entry, so it is >= 1 and its log is finite and >= 0.
    log_norm = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_norm
