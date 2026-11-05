from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def best_rank_k(A: ArrayLike, k: int) -> tuple[NDArray[np.float64], float]:
    """Return the best rank-k approximation of ``A`` and its Frobenius error.

    ``A`` is an (m, n) real array-like with m, n >= 1 and finite entries (integers allowed).
    ``k`` is an int with 0 <= k <= min(m, n); booleans are rejected. The result is a tuple
    (A_k, error): A_k is the float64 (m, n) matrix that keeps the k largest singular terms
    sum_{i<k} s_i u_i v_i^T, and error is a Python float equal to the square root of the sum of the
    discarded squared singular values, which is the Frobenius norm of A - A_k. A_k is the zero
    matrix when k = 0. Raise ValueError for invalid input. Do not mutate A.
    """
    raise NotImplementedError
