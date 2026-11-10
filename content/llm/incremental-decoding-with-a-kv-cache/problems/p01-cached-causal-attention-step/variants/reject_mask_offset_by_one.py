from __future__ import annotations

import math

import torch


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_cache(
    cache: tuple[torch.Tensor, torch.Tensor] | None, B: int, h: int, d_k: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Validate the cache and return (K_cache, V_cache, S). None means an empty cache."""
    if cache is None:
        return None, None, 0
    if not isinstance(cache, (tuple, list)) or len(cache) != 2:
        raise ValueError("cache must be None or a (K, V) pair")
    K_cache, V_cache = cache
    for name, t in (("K", K_cache), ("V", V_cache)):
        if not isinstance(t, torch.Tensor) or t.ndim != 4 or t.dtype != torch.float64:
            raise ValueError(f"cache {name} must be a 4-D float64 tensor (B, h, S, d_k)")
        if tuple(t.shape[:2]) != (B, h) or t.shape[3] != d_k:
            raise ValueError(f"cache {name} must have shape (B, h, S, d_k) for these heads")
    if tuple(K_cache.shape) != tuple(V_cache.shape):
        raise ValueError("cache K and V must have the same shape")
    return K_cache, V_cache, K_cache.shape[2]


def cached_attention_step(  # noqa: PLR0917 (the contract fixes seven arguments)
    x_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Wrong on purpose: one mistake, see the edit. Cached causal attention for T new tokens. See problem.md for the rules."""  # noqa: E501
    if not isinstance(x_new, torch.Tensor) or x_new.ndim != 3 or x_new.dtype != torch.float64:
        raise ValueError("x_new must be a 3-D float64 tensor of shape (B, T, d)")
    B, T, d = x_new.shape
    if B < 1 or T < 1:
        raise ValueError("x_new needs at least one sequence and one new token")
    if not _is_int(num_heads) or num_heads < 1 or d % num_heads:
        raise ValueError("num_heads must be a positive int that divides d")
    for name, W in (("W_q", W_q), ("W_k", W_k), ("W_v", W_v), ("W_o", W_o)):
        if not isinstance(W, torch.Tensor) or tuple(W.shape) != (d, d) or W.dtype != torch.float64:
            raise ValueError(f"{name} must be a float64 (d, d) tensor in nn.Linear layout")
    h = num_heads
    d_k = d // h

    # Read the cache. An empty cache is S = 0, and no concatenation is needed.
    K_cache, V_cache, S = _check_cache(cache, B, h, d_k)

    # Step 1, project only the new tokens: y = x @ W.T, shape (B, T, d). The cache holds the rest.
    Q = x_new @ W_q.T
    K_new = x_new @ W_k.T
    V_new = x_new @ W_v.T

    # Step 2, split into heads: (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k).
    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, T, h, d_k).transpose(1, 2)

    Qh, Kh_new, Vh_new = split(Q), split(K_new), split(V_new)

    # Step 3, append after the cache along the key axis (dim 2). cat builds new tensors, so the
    # input cache is never written.
    if K_cache is None or V_cache is None:  # _check_cache returns both as None or both set
        K_all, V_all = Kh_new, Vh_new
    else:
        K_all = torch.cat([K_cache, Kh_new], dim=2)  # (B, h, S + T, d_k)
        V_all = torch.cat([V_cache, Vh_new], dim=2)

    # Step 4, scores of each new query against every key: (B, h, T, S + T), scaled by 1/sqrt(d_k).
    scores = Qh @ K_all.transpose(-2, -1) / math.sqrt(d_k)

    # Step 5, bottom-right causal mask. Row i is the query at absolute position S + i, so it may
    # read key j when j <= S + i. tril(diagonal=S) on a (T, S + T) block encodes exactly that. With
    # S = 0 and T = 1 the mask is all True, so a single new token needs no mask.
    allowed = torch.ones(T, S + T, dtype=torch.bool).tril(diagonal=0)
    scores = scores.masked_fill(~allowed, float("-inf"))

    # Step 6, softmax over the keys, then the weighted sum of cached and new values.
    weights = torch.softmax(scores, dim=-1)
    context = weights @ V_all  # (B, h, T, d_k)

    # Step 7, merge heads back to (B, T, d) and apply the output projection.
    merged = context.transpose(1, 2).reshape(B, T, d)
    output = merged @ W_o.T

    return output, (K_all, V_all)
