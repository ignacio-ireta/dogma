from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cross_entropy_from_logits(
    logits: ArrayLike, targets: ArrayLike
) -> tuple[float, NDArray[np.float64]]:
    """Wrong on purpose: Return (mean cross-entropy, gradient w.r.t. logits) for integer class targets.

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
    """  # noqa: E501
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("logits must be a non-empty 2-D array with one row per example")
    n, k = z.shape
    if k < 2:
        raise ValueError("cross-entropy needs at least two classes per row")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite: NaN and infinity have no probability")

    t_raw = np.asarray(targets)
    if t_raw.ndim != 1 or t_raw.shape[0] != n:
        raise ValueError("targets must be 1-D with one class index per row of logits")
    if not np.issubdtype(t_raw.dtype, np.integer):
        raise ValueError("targets must be integer class indices")
    if t_raw.min() < 0 or t_raw.max() >= k:
        raise ValueError(f"targets must lie in [0, {k}) for {k} classes")
    t = t_raw.astype(np.intp)
    rows = np.arange(n)

    # Log-softmax from the logits. Subtracting the row maximum keeps every exponent at most 0.
    maxima = np.max(z, axis=1, keepdims=True)
    # An unrepresentable negative difference has zero probability. Keep its -inf limit
    # without a warning, while representable logits and gradients remain accurate.
    with np.errstate(over="ignore"):
        shifted = z - maxima
    e = np.exp(shifted)
    total = np.sum(e, axis=1, keepdims=True)
    log_total = np.log(total)

    # Divide before reducing, so a representable mean cannot overflow its intermediate sum.
    # If a true-class difference itself overflowed, divide its original terms first: the
    # batch mean may still be representable. Ordinary rows retain exact shifted differences.
    scaled_gap = -shifted[rows, t] / n
    outside_range = np.isinf(scaled_gap)
    with np.errstate(over="ignore"):
        scaled_gap[outside_range] = maxima[:, 0][outside_range] / n - z[rows, t][outside_range] / n
        loss = float(np.sum(scaled_gap + log_total[:, 0] / n))

    # Gradient: (softmax - onehot) / n. Subtracting one at the true class gives each row sum zero.
    grad = (e / total).astype(np.float32)
    grad[rows, t] -= 1.0
    grad /= n
    return loss, grad
