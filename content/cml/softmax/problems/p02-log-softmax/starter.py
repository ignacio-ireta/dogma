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
    raise NotImplementedError
