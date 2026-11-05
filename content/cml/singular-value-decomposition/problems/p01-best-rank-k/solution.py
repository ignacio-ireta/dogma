from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def best_rank_k(A: ArrayLike, k: int) -> tuple[NDArray[np.float64], float]:
    """Return the best rank-k approximation of ``A`` in the Frobenius norm, and its error.

    The singular value decomposition writes A as a sum of rank-one terms s_i u_i v_i^T, sorted by
    s_i. Keeping the first k terms gives the best rank-k matrix (Eckart-Young). The error of that
    truncation is the square root of the sum of the discarded squared singular values.
    """
    # np.array copies by default, so the caller's matrix is never modified.
    a = np.array(A, dtype=np.float64)

    if a.ndim != 2:
        raise ValueError(f"A must be 2-D, got {a.ndim}-D")
    m, n = a.shape
    if m < 1 or n < 1:
        raise ValueError("A must have at least one row and one column")
    if not np.isfinite(a).all():
        raise ValueError("A must contain only finite numbers (no NaN or inf)")
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    k = int(k)
    if not 0 <= k <= min(m, n):
        raise ValueError(f"k must be between 0 and min(m, n) = {min(m, n)}")

    # Economy SVD: a = U @ diag(s) @ Vt, with s sorted in decreasing order.
    u, s, vt = np.linalg.svd(a, full_matrices=False)

    # Keep the first k rank-one terms: A_k = sum_{i<k} s_i u_i v_i^T. Scaling the columns of U by s
    # and multiplying by the first k rows of Vt computes that sum in one product. For k = 0 both
    # factors have a zero dimension, and the product is the zero matrix.
    a_k = (u[:, :k] * s[:k]) @ vt[:k]

    # Eckart-Young identity: the Frobenius error of the truncation is the square root of the sum of
    # the discarded squared singular values. The sum is empty, hence zero, when k >= rank.
    error = float(np.sqrt(np.sum(s[k:] ** 2)))
    return a_k, error
