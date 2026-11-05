from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pseudo_inverse(A: ArrayLike, *, rcond: float = 1e-15) -> NDArray[np.float64]:
    """Return the Moore-Penrose pseudo-inverse of the (m, n) real matrix ``A`` as an (n, m) array.

    Compute it from the SVD, A = U diag(s) V^T. Every singular value s_i <= rcond * max(s) is
    treated as zero; the others are inverted, and the result is V diag(s_plus) U^T. The cutoff
    is relative to the largest singular value, as in numpy.linalg.pinv. The zero matrix gives
    the zero matrix.

    Raise ValueError for invalid input: not 2-D, empty, non-finite entries, or rcond that is not a
    finite number >= 0 (booleans rejected). Do not mutate A.

    Parameters
    ----------
    A : (m, n) real array-like, m, n >= 1, finite (integers allowed).
    rcond : keyword-only, finite real >= 0. Default 1e-15.
    """
    raise NotImplementedError
