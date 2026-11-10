from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def power_iteration(
    A: ArrayLike, num_iters: int = 1000, tol: float = 1e-10
) -> tuple[float, NDArray[np.float64]]:
    if isinstance(num_iters, bool) or not isinstance(num_iters, numbers.Integral) or num_iters < 1:
        raise ValueError("num_iters must be an integer >= 1")
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real) or not tol > 0:
        raise ValueError("tol must be a positive number")

    M = np.array(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < 1:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(M)):
        raise ValueError("matrix must be finite")
    scale = max(1.0, float(np.max(np.abs(M))))
    if float(np.max(np.abs(M - M.T))) > 1e-9 * scale:
        raise ValueError("matrix must be symmetric")

    def _norm(x: NDArray[np.float64]) -> float:
        return float(np.sqrt(x @ x))

    n = M.shape[0]
    # Fixed seeded start vector: generic, so it has a component along the dominant eigenvector.
    v = np.random.default_rng(0).standard_normal(n)
    v = v / _norm(v)

    lam = 0.0
    for _ in range(int(num_iters)):
        w = M @ v
        nw = _norm(w)
        if nw == 0.0:
            # M v = 0 for a unit v: (0, v) is an eigenpair.
            lam = 0.0
            break
        v = w / nw
        lam = float(v @ (M @ v))
        residual = M @ v - lam * v
        if _norm(residual) <= tol * max(1.0, abs(lam)):
            break

    absv = np.abs(v)
    idx = int(np.flatnonzero(absv >= (1.0 - 1e-9) * float(absv.max()))[0])
    if v[idx] < 0:
        v = -v
    return float(lam), np.asarray(v, dtype=np.float64)
