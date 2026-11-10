from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _checked_matrix(M: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.array(M, dtype=np.float64)  # copy
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite entries")
    return arr


def _unit_rows(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale each row to unit Euclidean length without overflow or underflow.

    Zero rows stay zero.
    """
    scale = np.max(np.abs(M), axis=1, keepdims=True)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    Y = M / safe_scale  # largest entry of each row is now 1 in magnitude
    norms = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return Y / safe_norms


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
    A_arr = _checked_matrix(A, "A")
    B_arr = _checked_matrix(B, "B")
    if A_arr.shape[1] != B_arr.shape[1]:
        raise ValueError(f"feature count mismatch: {A_arr.shape[1]} vs {B_arr.shape[1]}")

    S = _unit_rows(A_arr) @ _unit_rows(B_arr).T
    return np.clip(S, -1.0, 1.0).astype(np.float64)
