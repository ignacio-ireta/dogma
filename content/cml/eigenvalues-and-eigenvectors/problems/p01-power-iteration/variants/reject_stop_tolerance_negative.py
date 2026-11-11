from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SYMMETRY_RTOL = 1e-09
_TIE_RTOL = 1e-09


def power_iteration(
    A: ArrayLike, num_iters: int = 1000, tol: float = 1e-10
) -> tuple[float, NDArray[np.float64]]:
    """Return the dominant eigenpair of the real symmetric matrix ``A``.

    Repeatedly applying A tilts a vector toward the eigenvector whose eigenvalue has the largest
    magnitude. Each eigencomponent is multiplied by its eigenvalue, so the dominant one gains on the
    others at rate |λ2 / λ1| per step. Normalising every step stops the numbers from growing.
    """
    matrix = np.array(A, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("matrix must have at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite numbers")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if float(np.max(np.abs(matrix - matrix.T))) > _SYMMETRY_RTOL * scale:
        raise ValueError("matrix must be symmetric")
    if num_iters < 1:
        raise ValueError(f"num_iters must be at least 1, got {num_iters}")
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol}")
    n = int(matrix.shape[0])
    v = np.random.default_rng(0).standard_normal(n)
    v = v / np.sqrt(v @ v)
    eigenvalue = float(v @ (matrix @ v))
    for _ in range(num_iters):
        residual = matrix @ v - eigenvalue * v
        if np.sqrt(residual @ residual) <= tol * max(1.0, eigenvalue):
            break
        w = matrix @ v
        w_norm = float(np.sqrt(w @ w))
        if w_norm == 0.0:
            break
        v = w / w_norm
        eigenvalue = float(v @ (matrix @ v))
    magnitudes = np.abs(v)
    index = int(np.flatnonzero(magnitudes >= float(np.max(magnitudes)) * (1.0 - _TIE_RTOL))[0])
    if v[index] < 0:
        v = -v
    return (eigenvalue, v)
