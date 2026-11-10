from __future__ import annotations

import math

import torch


def scaled_dot_product_attention(  # noqa: PLR0912
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output, weights) for softmax(q @ k^T / sqrt(d_k)) @ v with an optional mask.

    q: floating (*batch, L, d_k); k: (*batch, S, d_k); v: (*batch, S, d_v), same dtype.
    mask: None or bool broadcastable to (*batch, L, S); True means the pair may attend.
    Weights are (*batch, L, S) with rows summing to 1 over allowed keys. A query row with no
    allowed key gives zero weights and zero output, never NaN. Raise ValueError on invalid input.
    Never mutate inputs; the result must be finite for large-magnitude scores.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k and v must have the same dtype")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dim() < 2:
            raise ValueError(f"{name} must have rank at least 2")
        if t.numel() == 0 or any(s == 0 for s in t.shape):
            raise ValueError(f"{name} must not have zero-sized axes")

    batch = q.shape[:-2]
    if k.shape[:-2] != batch or v.shape[:-2] != batch:
        raise ValueError("q, k and v must share the same batch shape")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same d_k")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same number of keys S")

    length = q.shape[-2]
    keys = k.shape[-2]
    target = (*batch, length, keys)

    allowed = None
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor")
        try:
            allowed = torch.broadcast_to(mask, target)
        except RuntimeError as exc:
            raise ValueError("mask cannot be broadcast to (*batch, L, S)") from exc

    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))

    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max)
    denom = exp_scores.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    weights = exp_scores / denom
    output = weights @ v
    return output, weights
