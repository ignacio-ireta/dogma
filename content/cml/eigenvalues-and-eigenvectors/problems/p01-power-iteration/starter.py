from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def power_iteration(
    A: ArrayLike, num_iters: int = 1000, tol: float = 1e-10
) -> tuple[float, NDArray[np.float64]]:
    """Return the dominant eigenpair (eigenvalue of largest |value|, unit eigenvector) of ``A``.

    Parameters
    ----------
    A : (n, n) real symmetric array-like, n >= 1 (ints allowed).
    num_iters : maximum number of power-iteration updates, >= 1.
    tol : stop when ||A v - λ v|| <= tol * max(1, |λ|), with λ = vᵀ A v, > 0.

    Returns
    -------
    (eigenvalue, eigenvector)
        eigenvalue : Python float, keeps its sign.
        eigenvector : (n,) float64 array of unit length. The first entry whose absolute value is
        within a relative 1e-9 of the largest absolute entry is positive.

    Raises
    ------
    ValueError
        If ``A`` is not square, not symmetric, or contains NaN or infinity, or if
        ``num_iters`` < 1 or ``tol`` <= 0.
    """
    raise NotImplementedError
