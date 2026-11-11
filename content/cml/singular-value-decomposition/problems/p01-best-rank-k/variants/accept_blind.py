from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def best_rank_k(A: ArrayLike, k: int) -> tuple[NDArray[np.float64], float]:
    try:
        M = np.array(A, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid A") from exc
    if M.ndim != 2 or M.size == 0 or not np.all(np.isfinite(M)):
        raise ValueError("invalid A")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("invalid k")
    k = int(k)
    if k < 0 or k > min(M.shape):
        raise ValueError("invalid k")
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
    err = float(np.sqrt(np.sum(s[k:] ** 2)))
    return np.asarray(Ak, dtype=np.float64).reshape(M.shape), err
