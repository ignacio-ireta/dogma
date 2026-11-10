from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def constrained_quadratic_min(
    Q: ArrayLike, c: ArrayLike, A: ArrayLike, b: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    Qm = np.array(Q, dtype=np.float64)
    cv = np.array(c, dtype=np.float64)
    Am = np.array(A, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)

    if Qm.ndim != 2 or Qm.shape[0] != Qm.shape[1] or Qm.shape[0] < 1:
        raise ValueError("Q must be a square matrix with at least one row")
    n = Qm.shape[0]
    if cv.shape != (n,):
        raise ValueError("c must have length n matching Q")
    if Am.ndim != 2 or Am.shape[0] < 1 or Am.shape[1] != n:
        raise ValueError("A must have shape (m, n) with m >= 1")
    m = Am.shape[0]
    if bv.shape != (m,):
        raise ValueError("b must have length m matching A")
    for arr in (Qm, cv, Am, bv):
        if not np.all(np.isfinite(arr)):
            raise ValueError("entries must be finite")

    scale = max(1.0, float(np.max(np.abs(Qm))))
    if float(np.max(np.abs(Qm - Qm.T))) > 1e-9 * scale:
        raise ValueError("Q must be symmetric")
    Qs = (Qm + Qm.T) / 2.0
    try:
        np.linalg.cholesky(Qs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Q must be positive definite") from exc

    if np.linalg.matrix_rank(Am) < m:
        raise ValueError("A is rank deficient: its rows are dependent")

    # Schur complement of the KKT system: Q x - A' lam = -c and A x = b.
    # With Q^{-1} applied via solves, S lam = b + A Q^{-1} c and x = Q^{-1}(A' lam - c).
    Qinv_c = np.linalg.solve(Qs, cv)
    Qinv_At = np.linalg.solve(Qs, Am.T)
    S = Am @ Qinv_At
    S = (S + S.T) / 2.0
    lam = np.linalg.solve(S, bv + Am @ Qinv_c)
    x = Qinv_At @ lam - Qinv_c
    return np.asarray(x, dtype=np.float64), np.asarray(lam, dtype=np.float64)
