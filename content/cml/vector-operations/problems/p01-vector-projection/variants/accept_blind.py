from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vector_projection(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the orthogonal projection of ``a`` onto ``b``.

    Parameters
    ----------
    a, b : 1-D array-likes of equal length n >= 1 (ints allowed).

    Returns
    -------
    (n,) float64 array equal to (a·b / b·b) · b.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` is not 1-D, their lengths differ, or ``b`` is the zero vector.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("a and b must be 1-D")
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same length")
    if a_arr.size == 0:
        raise ValueError("vectors must not be empty")

    b_max = np.max(np.abs(b_arr))
    if b_max == 0.0:
        raise ValueError("cannot project onto the zero vector")
    a_max = np.max(np.abs(a_arr))
    if a_max == 0.0:
        return np.zeros_like(a_arr)

    # Rescale both vectors so the dot products cannot overflow or underflow.
    # The projection is unchanged: (a·b / b·b) b = (a'·b' / b'·b') b' * a_max, with a' = a / a_max
    # and b' = b / b_max; the factor b_max cancels because the coefficient is scale-invariant in b.
    a_s = a_arr / a_max
    b_s = b_arr / b_max
    coeff = float(a_s @ b_s) / float(b_s @ b_s)
    return (coeff * b_s) * a_max
