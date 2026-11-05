from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def posterior(
    prior: ArrayLike, likelihood: ArrayLike, observations: ArrayLike
) -> NDArray[np.float64]:
    """Return the posterior P(H = i | observations) over h discrete hypotheses.

    Parameters
    ----------
    prior : (h,) probabilities, non-negative, summing to 1 within 1e-9.
    likelihood : (h, o) array; likelihood[i, j] = P(observation = j | hypothesis i).
        Each row sums to 1 within 1e-9.
    observations : 1-D sequence of integer category indices in [0, o). Observations are i.i.d.
        given the hypothesis. An empty sequence returns the prior.

    Returns
    -------
    (h,) float64 array summing to 1, computed in log space.

    Raises
    ------
    ValueError
        On bad shapes, negative or non-finite probabilities, probabilities that do not sum to 1,
        observation indices that are not integers or are out of range, and impossible
        observations (zero evidence).
    """
    raise NotImplementedError
