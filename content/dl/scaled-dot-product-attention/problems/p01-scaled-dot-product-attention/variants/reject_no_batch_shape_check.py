from __future__ import annotations

import math

import torch


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> None:
    """Check the contract before any arithmetic so each error names the broken rule."""
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise ValueError("q, k and v must be torch tensors")
    if not all(t.is_floating_point() for t in (q, k, v)):
        raise ValueError("q, k and v must be floating-point tensors")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k and v must share one dtype")
    if any(t.ndim < 2 for t in (q, k, v)):
        raise ValueError("q, k and v need at least two axes: (..., length, width)")
    if any(t.numel() == 0 for t in (q, k, v)):
        raise ValueError("zero-size axes are not allowed in attention inputs")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share the key width d_k")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must share the number of positions S")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor (True = may attend)")
        target = (*q.shape[:-2], q.shape[-2], k.shape[-2])
        try:
            torch.broadcast_to(mask, target)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to (*batch, L, S)") from exc


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrong on purpose: no batch-shape check, so mismatched batch axes broadcast silently."""
    _validate(q, k, v, mask)
    d_k = q.shape[-1]
    # Lesson step 1: scores z = q k^T / sqrt(d_k), shape (*batch, L, S). The division keeps
    # the variance of each dot product near 1 (Vaswani et al. 2017, section 3.2.1).
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    else:
        allowed = torch.broadcast_to(mask, scores.shape)
    # Lesson step 2: exclude disallowed pairs by setting their score to -inf before the softmax.
    masked = scores.masked_fill(~allowed, float("-inf"))
    # Lesson step 3: max-subtracted softmax over keys. Subtracting the row maximum leaves the
    # softmax unchanged and keeps every exponent at most 0. A fully masked row has maximum -inf,
    # so use 0 as its shift; then every exp is 0 and the row is handled by the guard below.
    row_max = masked.amax(dim=-1, keepdim=True)
    shift = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    expd = torch.exp(masked - shift)  # exp(-inf) = 0 for every disallowed pair
    denom = expd.sum(dim=-1, keepdim=True)
    # Rows with no allowed key have denom 0; give them zero weights rather than 0/0 = NaN.
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    weights = expd / safe_denom
    # Lesson step 4: output is the weighted average of the values, O = A V, shape (*batch, L, d_v).
    output = weights @ v
    return output, weights
