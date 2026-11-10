from __future__ import annotations

import math

import torch


def grouped_query_attention(  # noqa: PLR0917 (the contract fixes seven arguments)
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(X, torch.Tensor) or X.dim() != 3:
        raise ValueError("X must be a 3-D tensor")
    batch, length, d_model = X.shape
    for name, value in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int")
    if n_heads < n_kv_heads:
        raise ValueError("n_heads must be at least n_kv_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_kv_heads must divide n_heads")
    if d_model % n_heads != 0:
        raise ValueError("n_heads must divide d_model")
    d_head = d_model // n_heads
    group = n_heads // n_kv_heads
    kv_width = n_kv_heads * d_head

    expected = {
        "W_q": (d_model, d_model),
        "W_k": (kv_width, d_model),
        "W_v": (kv_width, d_model),
        "W_o": (d_model, d_model),
    }
    weights = {"W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o}
    for name, W in weights.items():
        if not isinstance(W, torch.Tensor) or tuple(W.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}")

    if mask is not None:
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (length, length)
        ):
            raise ValueError("mask must be a bool tensor of shape (L, L)")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every query row of the mask needs at least one True entry")

    q = (X @ W_q.T).reshape(batch, length, n_heads, d_head).transpose(1, 2)
    k = (X @ W_k.T).reshape(batch, length, n_kv_heads, d_head).transpose(1, 2)
    v = (X @ W_v.T).reshape(batch, length, n_kv_heads, d_head).transpose(1, 2)
    # Query head h reads key/value head h // group: repeat_interleave keeps the groups contiguous.
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_head)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights_sm = torch.softmax(scores, dim=-1)
    ctx = weights_sm @ v
    out = ctx.transpose(1, 2).reshape(batch, length, d_model) @ W_o.T
    return out
