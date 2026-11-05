from __future__ import annotations

from numpy.typing import ArrayLike


def determinant(A: ArrayLike) -> float:
    """Return the determinant of the square matrix ``A``.

    Uses LU-style elimination with partial pivoting: det = (sign of row swaps) * product of pivots.

    Parameters
    ----------
    A : (n, n) array-like, n >= 0 (ints allowed). The 0 x 0 matrix has determinant 1.0.

    Returns
    -------
    float, det(A). Large matrices may overflow or underflow the product of pivots.

    Raises
    ------
    ValueError
        If ``A`` is not 2-D, not square, or contains NaN or infinity.
    """
    raise NotImplementedError
