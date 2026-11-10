"""Wrong on purpose: temperature gets a default of 1.0, so it is no longer required."""

from __future__ import annotations

import math

import torch


def _check(a: object, b: object, temperature: object) -> None:
    """Validate everything before any arithmetic runs."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("a and b must be torch tensors")
    if a.ndim != 2 or not a.is_floating_point():
        raise ValueError("a must be a floating-point tensor of shape (N, D)")
    if b.shape != a.shape or b.dtype != a.dtype:
        raise ValueError("b must match a in shape (N, D) and dtype")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("need at least one pair and one embedding dimension")
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("embeddings must be finite")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a real number, not a bool")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Lesson step 1: divide each row by its Euclidean norm so a dot product becomes a cosine."""
    norms = torch.sqrt((x * x).sum(dim=1, keepdim=True))
    if bool((norms == 0).any()):
        raise ValueError("a zero embedding has no direction, so its cosine similarity is undefined")
    return x / norms


def _mean_cross_entropy_rows(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of each row against its diagonal entry, averaged over rows.

    Per row i: log sum_j exp(logits[i, j]) - logits[i, i]. The row maximum is subtracted
    before exponentiating, so the log-sum-exp cannot overflow when the logits are large.
    """
    n = logits.shape[0]
    row_max = logits.max(dim=1, keepdim=True).values  # (N, 1)
    shifted = torch.exp(logits - row_max).sum(dim=1)  # (N,), each term <= 1
    log_sum_exp = row_max.squeeze(1) + torch.log(shifted)  # (N,)
    diagonal = logits[torch.arange(n), torch.arange(n)]  # (N,), the positive pairs
    return (log_sum_exp - diagonal).mean()


def symmetric_infonce_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the symmetric InfoNCE loss as a 0-dim tensor with the dtype of a."""
    _check(a, b, temperature)
    a_hat = _l2_normalize(a)
    b_hat = _l2_normalize(b)

    # Lesson step 2: S[i, j] = cos(a_i, b_j) / temperature. The diagonal holds the positives.
    logits = (a_hat @ b_hat.T) / temperature  # (N, N)

    # Lesson step 3: cross-entropy along rows (a to b) and along columns (b to a). Transposing
    # the logits turns columns into rows, so the same helper serves both directions.
    loss_ab = _mean_cross_entropy_rows(logits)
    loss_ba = _mean_cross_entropy_rows(logits.T)

    # Lesson step 4: the symmetric loss is the average of the two directions.
    return (loss_ab + loss_ba) / 2
