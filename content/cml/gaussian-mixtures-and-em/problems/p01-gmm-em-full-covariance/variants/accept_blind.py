import numpy as np


def _floor(S, reg):
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(np.maximum(vals, reg)) @ vecs.T


def _loglik_terms(X, w, mu, cov):
    n, d = X.shape
    k = len(w)
    out = np.empty((n, k))
    for j in range(k):
        inv = np.linalg.inv(cov[j])
        _, logdet = np.linalg.slogdet(cov[j])
        diff = X - mu[j]
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        with np.errstate(divide="ignore"):
            lw = np.log(w[j])
        out[:, j] = lw - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    return out


def _e(X, w, mu, cov):
    T = _loglik_terms(X, w, mu, cov)
    m = T.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(T - m).sum(axis=1))
    return np.exp(T - lse[:, None]), float(lse.sum())


def _int(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(v)


def gmm_em(X, k, init_means, n_iter, reg):
    try:
        X = np.array(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be numeric") from exc
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D array with rows and columns")
    n, d = X.shape
    k = _int(k, "k")
    if not 1 <= k <= n:
        raise ValueError("k must be at least 1 and at most n")
    mu = np.array(init_means, dtype=np.float64)
    if mu.shape != (k, d) or not np.isfinite(mu).all():
        raise ValueError("init_means must be finite with shape (k, d)")
    n_iter = _int(n_iter, "n_iter")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    if isinstance(reg, bool) or not np.isfinite(float(reg)) or float(reg) <= 0:
        raise ValueError("reg must be a finite positive number")
    reg = float(reg)
    w = np.full(k, 1.0 / k)
    Xc = X - X.mean(axis=0)
    S0 = Xc.T @ Xc / n
    cov = np.stack([_floor(S0, reg) for _ in range(k)])
    lls = [_e(X, w, mu, cov)[1]]
    for _ in range(n_iter):
        R, _ = _e(X, w, mu, cov)
        Nk = R.sum(axis=0)
        new_w = np.zeros(k)
        new_mu = mu.copy()
        new_cov = cov.copy()
        for j in range(k):
            if Nk[j] == 0.0:
                continue
            new_w[j] = Nk[j] / n
            new_mu[j] = R[:, j] @ X / Nk[j]
            D = X - new_mu[j]
            new_cov[j] = _floor((D * R[:, j][:, None]).T @ D / Nk[j], reg)
        w, mu, cov = new_w, new_mu, new_cov
        lls.append(_e(X, w, mu, cov)[1])
    return w, mu, cov, np.array(lls)
