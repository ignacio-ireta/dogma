from __future__ import annotations

import math

import torch


def cached_attention_step(  # noqa: PLR0912, PLR0917
    x_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if not isinstance(x_new, torch.Tensor) or x_new.dtype != torch.float64 or x_new.dim() != 3:
        raise ValueError("x_new must be a 3-D float64 tensor")
    batch, chunk, d = x_new.shape
    if batch == 0 or chunk == 0:
        raise ValueError("B and T must be at least 1")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be a positive int")
    if d % num_heads != 0:
        raise ValueError("num_heads must divide d")
    h = num_heads
    dk = d // h
    for W in (W_q, W_k, W_v, W_o):
        if not isinstance(W, torch.Tensor) or W.dtype != torch.float64 or tuple(W.shape) != (d, d):
            raise ValueError("weights must be float64 (d, d) tensors")

    if cache is not None:
        if not isinstance(cache, (tuple, list)) or len(cache) != 2:
            raise ValueError("cache must be None or a (K, V) pair")
        k_c, v_c = cache
        for t in (k_c, v_c):
            if (
                not isinstance(t, torch.Tensor)
                or t.dtype != torch.float64
                or t.dim() != 4
                or t.shape[0] != batch
                or t.shape[1] != h
                or t.shape[3] != dk
            ):
                raise ValueError("cache tensors must be float64 (B, h, S, d_k)")
        if tuple(k_c.shape) != tuple(v_c.shape):
            raise ValueError("cache K and V must have the same shape")
    else:
        k_c = v_c = None

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch, chunk, h, dk).transpose(1, 2)

    q = split(x_new @ W_q.T)
    k_new = split(x_new @ W_k.T)
    v_new = split(x_new @ W_v.T)

    if k_c is None:
        past = 0
        keys = k_new
        vals = v_new
    else:
        past = k_c.shape[2]
        keys = torch.cat([k_c, k_new], dim=2)
        vals = torch.cat([v_c, v_new], dim=2)

    total = past + chunk
    scores = (q @ keys.transpose(-1, -2)) / math.sqrt(dk)
    j = torch.arange(total).reshape(1, total)
    i = torch.arange(chunk).reshape(chunk, 1)
    allowed = j <= past + i
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    ctx = weights @ vals
    out = ctx.transpose(1, 2).reshape(batch, chunk, d) @ W_o.T
    return out, (keys, vals)
