from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cosine_similarity_matrix(A: ArrayLike, B: ArrayLike) -> NDArray[np.float64]:
    """Return the (n, m) matrix of cosine similarities between rows of ``A`` and ``B``.

    Parameters
    ----------
    A : (n, d) array-like, n >= 1, d >= 1 (ints allowed).
    B : (m, d) array-like, m >= 1, same d.

    Returns
    -------
    (n, m) float64 array with S[i, j] = cos(A[i], B[j]), clipped to [-1, 1].
    Any pair involving a zero-norm row has similarity 0.0.

    Raises
    ------
    ValueError
        If an input is not 2-D, the column counts differ, an input is empty, or an
        entry is NaN or infinite.
    """
    raise NotImplementedError
