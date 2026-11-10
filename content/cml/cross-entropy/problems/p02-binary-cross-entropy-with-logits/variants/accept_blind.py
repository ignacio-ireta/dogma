from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def binary_cross_entropy_with_logits(z: ArrayLike, y: ArrayLike) -> float:
    """Return the mean binary cross-entropy of logits z against targets y in [0, 1].

    Uses the stable per-example form max(z, 0) - z * y + log(1 + exp(-|z|)), so that no
    probability is formed and no exponent is positive.

    Parameters
    ----------
    z : 1-D array-like of finite logits, length n >= 1.
    y : 1-D array-like of finite targets in [0, 1], the same length as z.

    Returns
    -------
    float: the mean loss over the n examples, never negative.

    Raises
    ------
    ValueError
        For non-1-D inputs, a length mismatch, an empty z, non-finite logits, or targets
        outside [0, 1].
    """
    z_arr = np.array(z, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    if z_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("z and y must be 1-D")
    if z_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("z and y must have the same length")
    n = z_arr.shape[0]
    if n == 0:
        raise ValueError("z must not be empty")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("logits must be finite")
    if not np.all((y_arr >= 0.0) & (y_arr <= 1.0)):
        raise ValueError("targets must lie in [0, 1]")

    with np.errstate(all="ignore"):
        # max(z, 0) - z * y equals z * (1 - y) for z >= 0 and -z * y for z < 0.
        # Using these forms avoids cancelling two large numbers, so the small
        # loss of a confident correct prediction keeps its relative precision.
        pos = z_arr >= 0.0
        linear = np.where(pos, z_arr * (1.0 - y_arr), -z_arr * y_arr)
        loss = linear + np.log1p(np.exp(-np.abs(z_arr)))
        # Divide before summing so the mean does not overflow when it is representable.
        mean = np.sum(loss / n)

    return float(max(mean, 0.0))
