from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def prior_shift_adjust(
    probs: Sequence[float] | np.ndarray,
    train_prior: float,
    deploy_prior: float,
) -> np.ndarray:
    """Probabilities of the positive class, corrected from a training to a deployment base rate.

    Under a prior shift the likelihood ratio is unchanged, so the odds move by
    r = [deploy_prior / (1 - deploy_prior)] / [train_prior / (1 - train_prior)].
    The corrected probability is r p / (r p + 1 - p). Probabilities of 0 and 1 stay 0 and 1.

    Returns
    -------
    NumPy float array of the same length as ``probs``.

    Raises
    ------
    ValueError
        If ``probs`` is not 1-D, holds a value that is NaN or outside [0, 1], or a prior is not
        strictly between 0 and 1.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1:
        raise ValueError("probs must be one-dimensional")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probs must be probabilities in [0, 1]")
    for name, prior in (("train_prior", train_prior), ("deploy_prior", deploy_prior)):
        # Written as a chained comparison so that NaN also fails the check.
        if not 0 < prior < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1")

    if train_prior == deploy_prior:
        return p.copy()

    # Multiplying odds becomes addition in log space. Forming the ratio directly can overflow
    # for valid priors near zero; log1p also preserves precision near either endpoint.
    log_ratio = (
        np.log(deploy_prior)
        - np.log1p(-deploy_prior)
        - np.log(train_prior)
        + np.log1p(-train_prior)
    )
    adjusted = p.copy()
    interior = (p > 0) & (p < 1)
    log_odds = np.log(p[interior]) - np.log1p(-p[interior]) + log_ratio
    # The sigmoid as exp(-log(1 + exp(-z))), with the log taken by logaddexp.
    adjusted[interior] = np.exp(-np.logaddexp(0.0, -log_odds))
    return adjusted
