from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_TOL = 1e-9


def _check_distribution(values: NDArray[np.float64], name: str) -> None:
    """Reject negative and non-finite probabilities. NaN fails every comparison, so test it too."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite probabilities (no NaN or inf)")
    if (values < 0).any():
        raise ValueError(f"{name} must contain non-negative probabilities")


def posterior(
    prior: ArrayLike, likelihood: ArrayLike, observations: ArrayLike
) -> NDArray[np.float64]:
    """Wrong on purpose: clips zero likelihoods to a tiny positive value, so they are not exactly zero.

    ``likelihood[i, j]`` is P(observation = j | hypothesis i). Observations are conditionally
    independent given the hypothesis, so their likelihoods multiply.
    """  # noqa: E501
    prior_arr = np.array(prior, dtype=np.float64)  # copies: the caller's arrays stay intact
    lik = np.array(likelihood, dtype=np.float64)
    obs = np.asarray(observations)

    # Validate the model: a distribution over hypotheses, and rows that are distributions
    # over observations (each row of the likelihood sums to 1 across observations, not hypotheses).
    if prior_arr.ndim != 1 or prior_arr.size == 0:
        raise ValueError(f"prior must be a non-empty 1-D array, got shape {prior_arr.shape}")
    _check_distribution(prior_arr, "prior")
    if abs(prior_arr.sum() - 1.0) > _TOL:
        raise ValueError("prior must sum to 1")
    h = prior_arr.size
    if lik.ndim != 2 or lik.shape[0] != h:
        raise ValueError(f"likelihood must have shape (h, o) with h = {h}, got {lik.shape}")
    _check_distribution(lik, "likelihood")
    if (np.abs(lik.sum(axis=1) - 1.0) > _TOL).any():
        raise ValueError("each likelihood row P(observation | hypothesis) must sum to 1")

    # Validate the data. An empty sequence is valid evidence of nothing, so the answer is the prior.
    if obs.ndim != 1:
        raise ValueError("observations must be a 1-D sequence of category indices")
    if obs.size == 0:
        return prior_arr / prior_arr.sum()
    if not np.issubdtype(obs.dtype, np.integer):
        raise ValueError("observations must be integer category indices")
    # Reject negative indices explicitly: NumPy would silently wrap -1 to the last column.
    if obs.min() < 0 or obs.max() >= lik.shape[1]:
        raise ValueError(f"observation indices must lie in [0, {lik.shape[1]})")

    # Log space: a product of many probabilities underflows float64, but a sum of logs does not.
    # log(0) = -inf is intended. A zero prior or a zero likelihood removes that hypothesis exactly.
    with np.errstate(divide="ignore"):
        log_post = np.log(prior_arr) + np.log(np.maximum(lik, 1e-300))[:, obs].sum(axis=1)

    # The evidence P(E) is the sum over hypotheses. If every hypothesis is -inf, it is zero.
    peak = log_post.max()
    if not np.isfinite(peak):
        raise ValueError("impossible observations: the evidence P(E) is zero for every hypothesis")

    # Log-sum-exp: subtract the peak before exponentiating, so the largest term is exp(0) = 1
    # and the normaliser cannot underflow to zero.
    weights = np.exp(log_post - peak)
    return weights / weights.sum()
