from __future__ import annotations

from numpy.typing import ArrayLike


def matrix_rank(A: ArrayLike, tol: float | None = None) -> int:
    """Return the rank of the 2-D matrix ``A`` by row reduction with partial pivoting.

    Parameters
    ----------
    A : (m, n) array-like, m >= 1, n >= 1 (ints allowed).
    tol : non-negative float, optional. A pivot with |p| <= tol counts as zero.
        Default: 1000 * max(m, n) * eps * ||A||_F.

    Returns
    -------
    int, the rank of A, with 0 <= rank <= min(m, n).

    Raises
    ------
    ValueError
        If ``A`` is not a non-empty 2-D matrix of finite numbers, or if ``tol`` is negative.
    """
    raise NotImplementedError
