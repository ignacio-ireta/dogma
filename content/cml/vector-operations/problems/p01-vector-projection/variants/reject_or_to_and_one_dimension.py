from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vector_projection(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the orthogonal projection of ``a`` onto ``b``: (a·b / b·b) · b."""
    a_vec = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)
    if a_vec.ndim != 1 and b_vec.ndim != 1:
        raise ValueError(f"expected 1-D vectors, got shapes {a_vec.shape} and {b_vec.shape}")
    if a_vec.shape != b_vec.shape:
        raise ValueError(f"length mismatch: {a_vec.shape[0]} vs {b_vec.shape[0]}")
    scale = float(np.max(np.abs(b_vec))) if b_vec.size else 0.0
    if scale == 0.0:
        raise ValueError("cannot project onto the zero vector")
    unit = b_vec / scale
    coefficient = float(a_vec @ unit) / float(unit @ unit)
    return coefficient * unit
