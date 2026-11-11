from __future__ import annotations

import numpy as np


def prior_shift_adjust(probs, train_prior, deploy_prior):
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1:
        raise ValueError("1-D")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("range")
    for prior in (train_prior, deploy_prior):
        if not 0 < prior < 1:
            raise ValueError("prior")
    out = p.copy()
    if train_prior == deploy_prior:
        return out
    # log-space odds to keep tiny priors finite
    lr = (np.log(deploy_prior) - np.log1p(-deploy_prior)) - (
        np.log(train_prior) - np.log1p(-train_prior)
    )
    for i, q in enumerate(p):
        if q == 0.0 or q == 1.0:  # noqa: PLR1714
            continue
        logit = np.log(q) - np.log1p(-q) + lr
        out[i] = 1.0 / (1.0 + np.exp(-logit))
    return out
