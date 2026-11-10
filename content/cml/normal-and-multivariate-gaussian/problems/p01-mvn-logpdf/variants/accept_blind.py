from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mvn_logpdf(X: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> NDArray[np.float64]:
    """Return the log-density of N(mean, cov) at each row of ``X``.

    Parameters
    ----------
    X : (n, d) array-like, n >= 1 rows and d >= 1 columns, all finite (ints allowed).
    mean : (d,) array-like, finite.
    cov : (d, d) array-like, symmetric and positive definite.

    Returns
    -------
    (n,) float64 array with
    -d/2 * log(2 pi) - 1/2 * log det(cov) - 1/2 * (x - mean)^T cov^{-1} (x - mean).
    Compute it with a Cholesky factorisation, never by forming the inverse or the determinant.

    Raises
    ------
    ValueError
        If X is not 2-D or is empty, mean does not have length d, cov is not (d, d), an input
        contains NaN or infinity, cov is not symmetric, or cov is not positive definite.
    """
    Xa = np.array(X, dtype=np.float64)
    mu = np.array(mean, dtype=np.float64)
    S = np.array(cov, dtype=np.float64)

    if Xa.ndim != 2 or Xa.shape[0] < 1 or Xa.shape[1] < 1:
        raise ValueError("X must be a non-empty 2-D array")
    n, d = Xa.shape  # noqa: RUF059
    if mu.ndim != 1 or mu.shape[0] != d:
        raise ValueError("mean must have length d")
    if S.shape != (d, d):
        raise ValueError("cov must have shape (d, d)")
    if not (np.all(np.isfinite(Xa)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(S))):
        raise ValueError("inputs must be finite")

    scale = max(1.0, float(np.max(np.abs(S))))
    if float(np.max(np.abs(S - S.T))) > 1e-10 * scale:
        raise ValueError("covariance must be symmetric")
    S = 0.5 * (S + S.T)

    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    # log det(S) = 2 * sum(log(diag(L)))
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))

    diff = (Xa - mu).T  # shape (d, n)
    z = np.linalg.solve(L, diff)  # L z = diff, so z^T z = diff^T S^{-1} diff
    maha = np.sum(z * z, axis=0)

    return (-0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det - 0.5 * maha).astype(np.float64)
