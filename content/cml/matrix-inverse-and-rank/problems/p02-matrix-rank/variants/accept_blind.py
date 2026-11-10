from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def matrix_rank(A: ArrayLike, tol: float | None = None) -> int:  # noqa: PLR0912
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
    arr = np.array(A, dtype=np.float64)  # copy
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
    m, n = arr.shape
    if m == 0 or n == 0:
        raise ValueError("expected a non-empty matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains NaN or infinite entries")

    if tol is None:
        scale = float(np.max(np.abs(arr)))
        if scale == 0.0:
            fro = 0.0
        else:
            fro = scale * float(np.sqrt(np.sum((arr / scale) ** 2)))
        tol = 1000 * max(m, n) * float(np.finfo(np.float64).eps) * fro
    else:
        tol = float(tol)
        if tol < 0:
            raise ValueError(f"tol must be non-negative; got {tol}")

    M = arr
    rank = 0
    for col in range(n):
        if rank == m:
            break
        idx = rank + int(np.argmax(np.abs(M[rank:, col])))
        if abs(M[idx, col]) <= tol:
            continue
        if idx != rank:
            M[[rank, idx]] = M[[idx, rank]]
        for i in range(rank + 1, m):
            M[i] -= (M[i, col] / M[rank, col]) * M[rank]
        rank += 1

    return int(rank)
