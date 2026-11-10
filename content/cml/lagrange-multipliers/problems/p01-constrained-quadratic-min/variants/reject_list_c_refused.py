from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _float_array(value: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    """Return a float64 copy of ``value`` with the required rank and only finite entries."""
    # np.array copies by default, so the caller's arrays are never written to.
    arr = np.array(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def constrained_quadratic_min(
    Q: ArrayLike, c: ArrayLike, A: ArrayLike, b: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrong on purpose: refuses c given as a list. Minimise 1/2 x'Qx + c'x subject to Ax = b, returning (x, lam).

    The Lagrangian is L = f(x) - lam'(Ax - b). This sign makes lam the derivative of the
    optimal value with respect to b. Stationarity of L in x gives Qx + c = A'lam, and
    stationarity in lam gives the constraints Ax = b. Together they form the KKT system

        [ Q   -A' ] [ x   ]   [ -c ]
        [ A    0  ] [ lam ] = [  b ]

    Raises
    ------
    ValueError
        On bad shapes, non-finite entries, an asymmetric or non-positive-definite Q, or a
        rank-deficient A.
    """  # noqa: E501
    Q_mat = _float_array(Q, "Q", ndim=2)
    if isinstance(c, list):
        raise ValueError("c must be an array")
    c_vec = _float_array(c, "c", ndim=1)
    A_mat = _float_array(A, "A", ndim=2)
    b_vec = _float_array(b, "b", ndim=1)

    n = Q_mat.shape[0]
    if n == 0 or Q_mat.shape != (n, n):
        raise ValueError(f"Q must be square with at least one row, got shape {Q_mat.shape}")
    if c_vec.shape != (n,):
        raise ValueError(f"c must have length {n} to match Q, got shape {c_vec.shape}")
    m = A_mat.shape[0]
    if m == 0 or A_mat.shape[1] != n:
        raise ValueError(f"A must have shape (m, {n}) with m >= 1, got {A_mat.shape}")
    if b_vec.shape != (m,):
        raise ValueError(f"b must have length {m} to match A, got shape {b_vec.shape}")

    # Symmetry is required of the Hessian Q. Compare with a tolerance relative to its scale,
    # so that rounding noise is accepted but a genuinely asymmetric matrix is not.
    scale = float(np.max(np.abs(Q_mat)))
    if float(np.max(np.abs(Q_mat - Q_mat.T))) > 1e-10 * scale:
        raise ValueError("Q must be symmetric")

    # Positive definiteness makes the objective strictly convex, so the KKT point is the
    # unique minimum. A Cholesky factorisation succeeds only for positive definite matrices.
    try:
        np.linalg.cholesky(Q_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Q must be positive definite") from exc

    # The KKT matrix is nonsingular only when A has full row rank: the constraints must be
    # independent. Check the rank explicitly, because a solve on a nearly dependent system
    # returns huge meaningless numbers instead of failing.
    if np.linalg.matrix_rank(A_mat) < m:
        raise ValueError("A is rank deficient: its rows are dependent")

    # Assemble the KKT block system. The -A' block comes from the -lam'(Ax - b) term in L.
    kkt = np.block([[Q_mat, -A_mat.T], [A_mat, np.zeros((m, m))]])
    rhs = np.concatenate([-c_vec, b_vec])
    solution = np.linalg.solve(kkt, rhs)

    # The first n entries are the minimiser, the last m are the multipliers.
    x = solution[:n].copy()
    lam = solution[n:].copy()
    return x, lam
