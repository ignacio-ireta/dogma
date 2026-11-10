from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _logits(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate a (n, k) logit matrix. -inf marks a class with zero probability."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "f":
        raise ValueError(f"{name} must be real numeric (bool, complex and object are rejected)")
    z = raw.astype(np.float64)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2-D array of shape (n, k)")
    if np.isnan(z).any() or np.isposinf(z).any():
        raise ValueError(f"{name} may contain -inf (masked class) but not NaN or +inf")
    if not np.isfinite(z).any(axis=1).all():
        raise ValueError(f"every row of {name} needs at least one finite logit")
    return z


def _log_softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """log softmax(z) along the last axis, stable for large and masked logits."""
    # Subtracting the row maximum (finite by the check above) keeps every exponent <= 0,
    # so exp cannot overflow. A masked logit stays -inf and exponentiates to exactly 0.
    shifted = z - np.max(z, axis=1, keepdims=True)
    # The sum includes the maximum's term, which is exactly 1, so the log is finite.
    log_norm = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_norm


def kl_divergence_from_logits(p_logits: ArrayLike, q_logits: ArrayLike) -> NDArray[np.float64]:
    """Wrong on purpose: integer inputs are rejected, only float inputs are accepted.

    KL = sum_j p_j (log p_j - log q_j). A term with p_j = 0 contributes 0 (0 log 0 = 0), even
    if q_j = 0. A term with p_j > 0 and q_j = 0 makes the row's KL +inf. Both inputs are (n, k),
    and the output is (n,) float64.
    """
    lp = _logits(p_logits, "p_logits")
    lq = _logits(q_logits, "q_logits")
    if lp.shape != lq.shape:
        raise ValueError("p_logits and q_logits must have the same (n, k) shape")

    log_p = _log_softmax(lp)
    log_q = _log_softmax(lq)

    # p_j > 0 exactly when its logit is finite. Test support from the logit, not from exp(log_p),
    # because exp can underflow to 0 while the class still has a positive probability.
    support = np.isfinite(log_p)
    q_zero = np.isneginf(log_q)
    with np.errstate(invalid="ignore", over="ignore"):
        # Where q_j > 0 both logs are finite, so this product is finite. Masked entries give
        # nan here, but np.where below never selects them.
        finite_term = np.exp(log_p) * (log_p - log_q)
    terms = np.where(support & q_zero, np.inf, np.where(support, finite_term, 0.0))
    return np.sum(terms, axis=1)
