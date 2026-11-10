from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def marginals_and_conditional(
    joint: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (p_x, p_y, p_y_given_x) for a joint probability table P[X=i, Y=j]."""
    try:
        table = np.array(joint, dtype=np.float64)  # copy: the input is never mutated
    except (TypeError, ValueError) as exc:
        raise ValueError("joint must be a rectangular numeric 2-D table") from exc
    if table.ndim != 2:
        raise ValueError("joint must be 2-D")
    if table.size == 0:
        raise ValueError("joint must not be empty")
    if not np.all(np.isfinite(table)):
        raise ValueError("joint entries must be finite")
    if np.any(table < 0):
        raise ValueError("joint entries must be non-negative")
    if abs(float(table.sum()) - 1.0) > 1e-9:
        raise ValueError("joint must sum to 1")

    p_x = table.sum(axis=1)
    p_y = table.sum(axis=0)
    cond = np.full(table.shape, np.nan, dtype=np.float64)
    defined = p_x > 0
    cond[defined] = table[defined] / p_x[defined, None]
    return p_x, p_y, cond
