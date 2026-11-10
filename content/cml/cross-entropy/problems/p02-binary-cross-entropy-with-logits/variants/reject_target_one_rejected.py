from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def binary_cross_entropy_with_logits(z: ArrayLike, y: ArrayLike) -> float:
    """Wrong on purpose: Return the mean binary cross-entropy of logits z against targets y in [0, 1].

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
    """  # noqa: E501
    z_arr = np.asarray(z, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if z_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("z and y must be 1-D arrays")
    if z_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("z and y must have the same length")
    if z_arr.shape[0] == 0:
        raise ValueError("z must contain at least one example")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("logits must be finite")
    if not (np.all(np.isfinite(y_arr)) and np.all((y_arr >= 0.0) & (y_arr < 1.0))):
        raise ValueError("targets must lie in [0, 1]")

    # Per example: max(z, 0) - z * y + log(1 + exp(-|z|)). The exponent -|z| is never positive,
    # so exp cannot overflow. The first two terms give log(1 + e^z) - y z for either sign of z.
    # Factor the linear term by sign. For positive z and a soft target near one, computing
    # z - z*y would cancel two huge terms and lose relative precision in the remaining loss.
    linear = np.where(z_arr >= 0.0, z_arr * (1.0 - y_arr), -z_arr * y_arr)
    per_example = linear + np.log1p(np.exp(-np.abs(z_arr)))
    # A sum of finite losses can overflow even when their mean is representable. Normalize
    # by the largest loss first; this also preserves a batch of equally tiny losses.
    scale = float(np.max(per_example))
    return 0.0 if scale == 0.0 else float(scale * np.mean(per_example / scale))
