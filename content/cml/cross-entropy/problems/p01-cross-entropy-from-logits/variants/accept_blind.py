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
    z = np.asarray(logits)
    if z.ndim != 2 or z.shape[0] < 1:
        raise ValueError("logits must be a non-empty 2-D array")
    n, k = z.shape
    if k < 2:
        raise ValueError("logits need at least two classes")
    z = z.astype(np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")

    t = np.asarray(targets)
    if t.ndim != 1 or t.shape[0] != n:
        raise ValueError("targets must be a 1-D array with one entry per row of logits")
    if t.dtype.kind not in "iu":
        raise ValueError("targets must be integer class indices")
    if t.min() < 0 or t.max() >= k:
        raise ValueError("every target must lie in [0, k)")
    t = t.astype(np.intp)
    rows = np.arange(n)

    with np.errstate(over="ignore"):
        m = z.max(axis=1, keepdims=True)
        d = z - m  # <= 0; overflow to -inf maps to probability 0 correctly
    e = np.exp(d)
    s = np.sum(e, axis=1)

    # Per-row loss = max - z_t + log(sum_j exp(z_j - max)), each divided by n before summing,
    # so that the mean stays representable even when a single logit gap would overflow.
    m_vec = m[:, 0]
    zt = z[rows, t]
    with np.errstate(over="ignore"):
        term = m_vec / n - zt / n + np.log(s) / n
    loss = float(np.sum(term))

    grad = e / s[:, None]
    grad[rows, t] -= 1.0
    grad /= n
    return loss, grad
