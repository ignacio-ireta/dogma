from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vector_projection(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the orthogonal projection of ``a`` onto ``b``: (a·b / b·b) · b."""
    # Convert to float64 copies: ints can't hold the fractional coefficient, and we
    # must never write into the caller's arrays.
    a_vec = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)

    if a_vec.ndim != 1 or b_vec.ndim != 1:
        raise ValueError(f"expected 1-D vectors, got shapes {a_vec.shape} and {b_vec.shape}")
    if a_vec.shape != b_vec.shape:
        raise ValueError(f"length mismatch: {a_vec.shape[0]} vs {b_vec.shape[0]}")

    # Zero means there is no line to project onto. Test the largest entry rather than b·b:
    # for a tiny but nonzero b, b·b underflows to 0, and for a huge b it overflows to inf.
    scale = float(np.max(np.abs(b_vec))) if b_vec.size else 0.0
    if scale == 0.0:
        raise ValueError("cannot project onto the zero vector")

    # Projecting onto b and onto u = b / scale gives the same point, because both lie on the
    # same line. Every entry of u is at most 1 in magnitude, so u·u lies in [1, n] and cannot
    # overflow or underflow. The coefficient c makes the residual (a - c·u) orthogonal to u.
    unit = b_vec / scale
    direction = unit / np.sqrt(float(unit @ unit))
    return float(a_vec @ direction) * direction
