from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _check_distribution(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite")
    if np.any(values < 0):
        raise ValueError(f"{label} must be non-negative")


def posterior(
    prior: ArrayLike, likelihood: ArrayLike, observations: ArrayLike
) -> NDArray[np.float64]:
    """Return the posterior P(H = i | observations), computed in log space."""
    pr = np.array(prior, dtype=np.float64)
    lk = np.array(likelihood, dtype=np.float64)
    if pr.ndim != 1 or pr.shape[0] < 1:
        raise ValueError("prior must be a non-empty 1-D array")
    if lk.ndim != 2 or lk.shape[1] < 1:
        raise ValueError("likelihood must be a 2-D array with at least one column")
    if lk.shape[0] != pr.shape[0]:
        raise ValueError("prior and likelihood disagree in the number of hypotheses")
    _check_distribution(pr, "prior")
    _check_distribution(lk, "likelihood")
    if abs(float(pr.sum()) - 1.0) > 1e-9:
        raise ValueError("prior must sum to 1")
    if np.any(np.abs(lk.sum(axis=1) - 1.0) > 1e-9):
        raise ValueError("each likelihood row must sum to 1")

    n_obs_values = lk.shape[1]
    obs_list = list(observations)
    indices = []
    for o in obs_list:
        if isinstance(o, bool) or not isinstance(o, (int, np.integer)):
            raise ValueError("observations must be integer indices")
        if o < 0 or o >= n_obs_values:
            raise ValueError("observation index out of range")
        indices.append(int(o))

    if not indices:
        return pr.copy()

    counts = np.bincount(np.array(indices, dtype=np.intp), minlength=n_obs_values)
    counts_f = counts.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pr = np.log(pr)
        log_lk = np.log(lk)
        # Only categories that were observed contribute; 0 * -inf is avoided by np.where.
        terms = np.where(counts[None, :] > 0, counts_f[None, :] * log_lk, 0.0)
        log_w = log_pr + terms.sum(axis=1)

    m = float(np.max(log_w))
    if not np.isfinite(m):
        raise ValueError("impossible observations: the evidence is zero")
    with np.errstate(under="ignore"):
        w = np.exp(log_w - m)
    return np.asarray(w / w.sum(), dtype=np.float64)
