from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cross_entropy_from_logits(
    logits: ArrayLike, targets: ArrayLike
) -> tuple[float, NDArray[np.float64]]:
    """Return (mean cross-entropy, gradient w.r.t. logits) for integer class targets.

    The loss is -(1/n) sum_i log softmax(logits_i)[targets_i], computed from the logits.
    The gradient of row i is (softmax(logits_i) - onehot(targets_i)) / n, so each row sums to zero.

    Parameters
    ----------
    logits : (n, k) array-like of finite values, n >= 1, k >= 2.
    targets : (n,) array of integer class indices in [0, k).

    Returns
    -------
    (loss, grad): a float and a float64 array of shape (n, k).

    Raises
    ------
    ValueError
        For bad shapes, k < 2, non-finite logits, non-integer or out-of-range targets.
    """
    raise NotImplementedError
