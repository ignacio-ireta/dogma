from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _floor_covariance(scatter: NDArray[np.float64], reg: float) -> NDArray[np.float64]:
    """Return the nearest matrix with every eigenvalue at least ``reg`` (the constrained M-step)."""
    # Raising each eigenvalue to max(lambda, reg) keeps the eigenvectors. It is the exact
    # maximiser of the EM objective over covariances with smallest eigenvalue >= reg, which is
    # why EM stays monotone. Adding reg * I instead is a penalty, not the constrained maximiser.
    values, vectors = np.linalg.eigh(scatter)
    return (vectors * np.maximum(values, reg)) @ vectors.T


def _log_joint(
    X: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the (n, k) matrix of log(pi_k) + log N(x_i | mu_k, Sigma_k), in log space."""
    n, d = X.shape
    k = weights.shape[0]
    out = np.empty((n, k))
    with np.errstate(divide="ignore"):  # a zero weight gives -inf, which is the intended value
        log_w = np.log(weights)
    for j in range(k):
        # Cholesky factor L with Sigma = L L'. Differences are formed directly as X - mu, never
        # through |x|^2 - 2x.mu + |mu|^2, so a large common offset does not lose precision.
        L = np.linalg.cholesky(covs[j])
        z = np.linalg.solve(L, (X - means[j]).T)  # z = L^{-1}(x - mu), column per sample
        mahalanobis = np.sum(z * z, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_normal = -0.5 * (d * np.log(2.0 * np.pi) + log_det + mahalanobis)
        out[:, j] = log_w[j] + log_normal
    return out


def _e_step(
    X: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return responsibilities r_ik = P(z_i = k | x_i) and the current log-likelihood."""
    log_joint = _log_joint(X, weights, means, covs)
    # log-sum-exp over components: subtract each row's maximum before exponentiating.
    row_max = log_joint.max(axis=1, keepdims=True)
    log_norm = row_max + np.log(np.exp(log_joint - row_max).sum(axis=1, keepdims=True))
    resp = np.exp(log_joint - log_norm)
    return resp, float(log_norm.sum())


def _as_samples(X: ArrayLike) -> NDArray[np.float64]:
    """Return a float64 copy of the samples, checked to be a finite (n, d) array with n, d >= 1."""
    try:
        X_arr = np.array(X, dtype=np.float64)  # copies, so the caller's array is never written to
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a 2-D array of numbers") from exc
    if X_arr.ndim != 2 or X_arr.shape[0] < 1 or X_arr.shape[1] < 1:
        raise ValueError(f"X must be a 2-D array with at least one sample, got {X_arr.shape}")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must contain only finite numbers")
    return X_arr


def _check_count(value: object, name: str) -> int:
    """Return an integer count, rejecting booleans and non-integers."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, not a boolean or a float")
    return int(value)


def _check_floor(reg: object) -> float:
    """Return the covariance floor, which must be a finite number strictly above zero."""
    if isinstance(reg, bool):
        raise ValueError("reg must be a real number, not a boolean")
    try:
        reg_value = float(reg)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("reg must be a real number") from exc
    if not np.isfinite(reg_value):
        raise ValueError("reg must be a finite number")
    if reg_value <= 0.0:
        raise ValueError("reg must be strictly positive: a zero floor allows collapse")
    return reg_value


def _validate(
    X: ArrayLike, k: int, init_means: ArrayLike, n_iter: int, reg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, float]:
    """Return validated float64 samples, initial means, the iteration count and the floor."""
    X_arr = _as_samples(X)
    n, d = X_arr.shape
    k_value = _check_count(k, "k")
    if not 1 <= k_value <= n:
        raise ValueError("k must be at least 1 and at most the number of samples")

    means = np.array(init_means, dtype=np.float64)
    if means.shape != (k_value, d):
        raise ValueError(f"init_means must have shape (k, d) = {(k_value, d)}, got {means.shape}")
    if not np.all(np.isfinite(means)):
        raise ValueError("init_means must contain only finite numbers")

    iterations = _check_count(n_iter, "n_iter")
    if iterations < 0:
        raise ValueError("n_iter must be non-negative")
    return X_arr, means, iterations, _check_floor(reg)


def gmm_em(
    X: ArrayLike,
    k: int,
    init_means: ArrayLike,
    n_iter: int,
    reg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit a full-covariance Gaussian mixture with EM, returning (weights, means, covs, lls).

    Initial parameters: equal weights 1/k, the given means, and every covariance equal to the
    pooled covariance of X (divided by n) with its eigenvalues floored at ``reg``. Each iteration
    runs an E-step (responsibilities) and an M-step (weighted maximum likelihood, then the floor).
    ``lls[t]`` is the log-likelihood after t iterations, so ``lls`` has length ``n_iter + 1``.
    """
    X_arr, means, n_iter, reg_value = _validate(X, k, init_means, n_iter, reg)
    n = X_arr.shape[0]
    k = means.shape[0]
    weights = np.full(k, 1.0 / k)

    # Initial covariance: the pooled covariance, floored. Every component starts identical.
    centred = X_arr - X_arr.mean(axis=0)
    pooled = centred.T @ centred / n
    covs = np.array([_floor_covariance(pooled, reg_value) for _ in range(k)])

    lls = [_e_step(X_arr, weights, means, covs)[1]]
    for _ in range(n_iter):
        resp, _ = _e_step(X_arr, weights, means, covs)

        # M-step. N_k is the effective number of samples in component k.
        counts = resp.sum(axis=0)
        new_weights = counts / n
        new_means = means.copy()
        new_covs = covs.copy()
        for j in range(k):
            if counts[j] == 0.0:
                # No responsibility at all: the component is dead. Keep its mean and covariance
                # and give it weight 0, so no division by N_k = 0 happens.
                new_weights[j] = 0.0
                continue
            new_means[j] = resp[:, j] @ X_arr / counts[j]  # weighted mean, using the new mean
            diff = X_arr - new_means[j]
            scatter = (resp[:, j, None] * diff).T @ diff / counts[j]
            new_covs[j] = _floor_covariance(scatter, reg_value)

        weights, means, covs = new_weights, new_means, new_covs
        lls.append(_e_step(X_arr, weights, means, covs)[1])

    return weights, means, covs, np.array(lls, dtype=np.float64)


_correct = gmm_em


def _change_in_place(v):
    if v.dtype.kind == "f":
        v[...] = v.round(6)


def gmm_em(*args, **kwargs):
    """Wrong on purpose: the result is right, but this rounds `init_means` in place."""
    result = _correct(*args, **kwargs)
    target = args[2] if len(args) > 2 else kwargs.get("init_means")
    if type(target).__name__ in ("ndarray",):
        _change_in_place(target)
    return result


gmm_em.__wrapped__ = _correct
