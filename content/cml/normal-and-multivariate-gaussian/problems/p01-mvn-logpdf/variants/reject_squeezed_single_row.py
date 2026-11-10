from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mvn_logpdf(X: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: return of N(mean, cov) at each row of ``X``.

    Factor cov = L L^T (Cholesky). Then log det(cov) = 2 sum(log diag(L)), and the quadratic
    form (x - mean)^T cov^{-1} (x - mean) equals |z|^2 where L z = (x - mean).
    """
    x = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(cov, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"expected X with shape (n, d), got {x.ndim}-D")
    n, d = x.shape
    if n == 0 or d == 0:
        raise ValueError("need at least one observation and one variable")
    if mu.shape != (d,):
        raise ValueError(f"mean must have length {d}, got shape {mu.shape}")
    if sigma.shape != (d, d):
        raise ValueError(f"covariance must have shape ({d}, {d}), got {sigma.shape}")
    if not (np.isfinite(x).all() and np.isfinite(mu).all() and np.isfinite(sigma).all()):
        raise ValueError("inputs must be finite (no NaN or inf)")
    # Cholesky reads only one triangle, so an asymmetric matrix would be silently replaced
    # by a different one. Check symmetry first, with a tolerance scaled to the matrix.
    scale = max(1.0, float(np.max(np.abs(sigma))))
    if np.max(np.abs(sigma - sigma.T)) > 1e-10 * scale:
        raise ValueError("covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    # Sum of logs of the factor's diagonal. The determinant itself would leave float64 range
    # at moderate d; the sum of logs never does.
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    # Solve L z = (x - mean) for every row at once. Then (x - mean)^T cov^{-1} (x - mean) = |z|^2,
    # so no inverse is formed.
    diff = (x - mu).T
    z = np.linalg.solve(chol, diff)
    mahalanobis = np.sum(z * z, axis=0)
    return np.squeeze(-0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det - 0.5 * mahalanobis)
