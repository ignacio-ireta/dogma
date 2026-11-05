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
    raise NotImplementedError
