from __future__ import annotations

from numpy.typing import ArrayLike


def vector_norm(x: ArrayLike, p: float) -> float:
    """Return the p-norm of a 1-D vector x, for real p >= 1 or p = inf.

    Parameters
    ----------
    x : 1-D array-like of finite real numbers, length n >= 1 (ints allowed).
    p : real number >= 1, or ``np.inf``.

    Returns
    -------
    float >= 0: (sum_i |x_i|**p) ** (1/p), or max_i |x_i| when p is infinite.
    Must not overflow or underflow for entries anywhere in the float64 range.

    Raises
    ------
    ValueError
        If p is below 1 or NaN, if x is not 1-D, or if x is empty.
    """
    raise NotImplementedError
