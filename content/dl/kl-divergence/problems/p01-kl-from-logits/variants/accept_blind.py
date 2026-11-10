from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _prepare(name: str, logits: ArrayLike) -> NDArray[np.float64]:
    try:
        arr = np.asarray(logits)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be array-like") from exc
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must hold real numeric values, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D with shape (n, k)")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    arr = arr.astype(np.float64)
    if np.isnan(arr).any() or np.isposinf(arr).any():
        raise ValueError(f"{name} must not contain NaN or +inf")
    if not np.isfinite(arr).any(axis=1).all():
        raise ValueError(f"every row of {name} needs at least one finite entry")
    return arr


def _log_softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.max(x, axis=1, keepdims=True)  # finite, since each row has a finite entry
    with np.errstate(divide="ignore"):
        lse = m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))
        return x - lse


def kl_divergence_from_logits(p_logits: ArrayLike, q_logits: ArrayLike) -> NDArray[np.float64]:
    """Return KL(P_i || Q_i) in nats for each row i, with P_i = softmax(p_logits[i]).

    KL = sum_j p_j (log p_j - log q_j). A term with p_j = 0 contributes 0 (0 log 0 = 0), even if
    q_j = 0. A term with p_j > 0 and q_j = 0 makes the row's KL +inf. Both inputs are (n, k) with
    n >= 1, k >= 1; -inf logits are allowed (zero probability); NaN and +inf are not. Every row
    needs a finite entry. Returns a float64 array of shape (n,). Raise ValueError for invalid
    input. Compute the log-probabilities with log-sum-exp from the logits. Do not mutate inputs.
    """
    p = _prepare("p_logits", p_logits)
    q = _prepare("q_logits", q_logits)
    if p.shape != q.shape:
        raise ValueError("p_logits and q_logits must have the same shape")

    log_p = _log_softmax(p)
    log_q = _log_softmax(q)
    with np.errstate(invalid="ignore", over="ignore"):
        p_prob = np.exp(log_p)
        diff = log_p - log_q
        # A term is included only when p_j > 0; this avoids 0 * inf = nan for tiny p_j.
        terms = np.where(p_prob > 0, p_prob * diff, 0.0)
        kl = np.sum(terms, axis=1)
    finite = np.isfinite(kl)
    kl = np.where(finite, np.maximum(kl, 0.0), kl)
    return kl.astype(np.float64)
