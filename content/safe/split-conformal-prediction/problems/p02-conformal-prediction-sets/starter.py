from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def prediction_sets(
    cal_probs: NDArray[np.float64],
    cal_labels: NDArray[np.intp],
    test_probs: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.bool_]:
    """Return a boolean (m, K) mask of classes in each conformal prediction set.

    cal_probs: (n, K) calibration probabilities, rows sum to 1, n >= 1, K >= 2.
    cal_labels: (n,) integer true labels in [0, K).
    test_probs: (m, K) test probabilities with the same K, m >= 0.
    alpha: float in (0, 1). Score = 1 - p(true class). Threshold q is the k-th smallest
    calibration score with k = ceil((n + 1)(1 - alpha)) computed exactly, or +inf when k > n.
    out[i, j] is True when 1 - test_probs[i, j] <= q. Raise ValueError for invalid arguments.
    """
    raise NotImplementedError
