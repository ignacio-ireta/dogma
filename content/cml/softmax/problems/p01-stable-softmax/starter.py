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
    raise NotImplementedError
