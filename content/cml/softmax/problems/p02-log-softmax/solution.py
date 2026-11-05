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
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 0:
        raise ValueError("log_softmax needs at least one dimension to normalise over")
    if not np.all(np.isfinite(z)):
        raise ValueError("log_softmax needs finite logits: NaN and infinity have no distribution")
    try:
        length = z.shape[axis]
    except IndexError as exc:
        raise ValueError(f"axis {axis} is out of range for {z.ndim} dimension(s)") from exc
    if length == 0:
        raise ValueError("log_softmax needs at least one logit along the normalised axis")

    # Step 1: the per-slice maximum, kept as an axis so it broadcasts back onto its own slice.
    m = np.max(z, axis=axis, keepdims=True)
    # Only an unrepresentable negative difference can overflow for finite inputs. Keep its
    # -inf limit without warning; its exponential is zero.
    with np.errstate(over="ignore"):
        shifted = z - m

    # Step 2: log normalizer in shifted coordinates. Adding m back can erase this small
    # correction when the common logit offset is huge, even for a uniform distribution.
    log_norm = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    # Step 3: log softmax_j = z_j - log-sum-exp. Subtracting directly keeps tiny probabilities
    # finite: a probability that would round to 0 still has a finite logarithm here.
    return shifted - log_norm
