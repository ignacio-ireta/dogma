from __future__ import annotations

import math

import torch


def multi_head_self_attention(  # noqa: PLR0917 (the contract fixes eleven arguments)
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    b_q: torch.Tensor,
    b_k: torch.Tensor,
    b_v: torch.Tensor,
    b_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the multi-head self-attention output for a batch of sequences."""
    if X.dim() != 3:
        raise ValueError("X must be 3-D (B, L, d_model)")
    B, L, d = X.shape
    for name, W in (("W_q", W_q), ("W_k", W_k), ("W_v", W_v), ("W_o", W_o)):
        if tuple(W.shape) != (d, d):
            raise ValueError(f"{name} must have shape (d_model, d_model)")
    for name, b in (("b_q", b_q), ("b_k", b_k), ("b_v", b_v), ("b_o", b_o)):
        if tuple(b.shape) != (d,):
            raise ValueError(f"{name} must have shape (d_model,)")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if d % num_heads != 0:
        raise ValueError("num_heads must divide d_model")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("mask must be a torch.bool tensor")
        if tuple(mask.shape) != (L, L):
            raise ValueError("mask must have shape (L, L)")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every query row of mask needs at least one True entry")

    H = num_heads
    dk = d // H
    q = X @ W_q.T + b_q
    k = X @ W_k.T + b_k
    v = X @ W_v.T + b_v

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)

    q, k, v = split(q), split(k), split(v)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dk)  # (B, H, L, L)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    heads = attn @ v  # (B, H, L, dk)
    merged = heads.transpose(1, 2).reshape(B, L, d)
    return merged @ W_o.T + b_o
