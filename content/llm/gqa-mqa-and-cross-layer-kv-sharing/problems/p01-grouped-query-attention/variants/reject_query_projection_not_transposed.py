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
    """Wrong on purpose: uses X @ W_q instead of X @ W_q.T. Grouped-query self-attention, written out step by step (Ainslie et al. 2023).

    Shapes: X (B, L, d); query projection (B, L, d); key and value projections
    (B, L, n_kv * d_head); query heads (B, h, L, d_head); key and value heads
    (B, n_kv, L, d_head); scores (B, h, L, L).
    """  # noqa: E501
    # Validate the interface first. Each rule is a contract rule, so a wrong shape never
    # broadcasts silently.
    if X.ndim != 3:
        raise ValueError("X must be 3-D (B, L, d_model)")
    B, L, d = X.shape
    for name, value in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n_heads % n_kv_heads:
        raise ValueError("n_kv_heads must divide n_heads")
    if d % n_heads:
        raise ValueError("n_heads must divide d_model")

    d_head = d // n_heads
    kv_width = n_kv_heads * d_head  # the key and value projections are narrower than d_model
    expected = {"W_q": (d, d), "W_k": (kv_width, d), "W_v": (kv_width, d), "W_o": (d, d)}
    for name, W in (("W_q", W_q), ("W_k", W_k), ("W_v", W_v), ("W_o", W_o)):
        if tuple(W.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]} in nn.Linear layout")
    if mask is not None:
        if mask.dtype != torch.bool or tuple(mask.shape) != (L, L):
            raise ValueError("mask must be a torch.bool tensor of shape (L, L)")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every query row of the mask needs at least one allowed key")

    # Step 1, projections. Queries use all n_heads heads. Keys and values use only n_kv_heads heads,
    # so their projections write kv_width features instead of d_model.
    Q = X @ W_q  # (B, L, d)
    K = X @ W_k.T  # (B, L, n_kv * d_head)
    V = X @ W_v.T  # (B, L, n_kv * d_head)

    # Step 2, split heads. The head axis moves in front of the token axis, as in MHA.
    Qh = Q.reshape(B, L, n_heads, d_head).transpose(1, 2)  # (B, h, L, d_head)
    Kh = K.reshape(B, L, n_kv_heads, d_head).transpose(1, 2)  # (B, n_kv, L, d_head)
    Vh = V.reshape(B, L, n_kv_heads, d_head).transpose(1, 2)  # (B, n_kv, L, d_head)

    # Step 3, the group mapping. Each query head h reads key/value head h // group_size.
    # repeat_interleave repeats each key/value head group_size times in a row, so the copies
    # line up as [kv0, kv0, kv1, kv1, ...]. That is exactly h // group_size.
    # (repeat would give h % n_kv_heads.)
    group_size = n_heads // n_kv_heads
    Kh = Kh.repeat_interleave(group_size, dim=1)  # (B, h, L, d_head)
    Vh = Vh.repeat_interleave(group_size, dim=1)  # (B, h, L, d_head)

    # Step 4, scores per query head: Qh @ Kh^T / sqrt(d_head). The scale uses the head width.
    scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(d_head)  # (B, h, L, L)
    if mask is not None:
        # True means allowed. Blocked pairs get -inf, so their softmax weight is exactly zero.
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 5, softmax over keys, then the weighted sum of values per query head.
    weights = torch.softmax(scores, dim=-1)  # (B, h, L, L)
    context = weights @ Vh  # (B, h, L, d_head)

    # Step 6, merge heads back to (B, L, d), then apply the output projection.
    merged = context.transpose(1, 2).reshape(B, L, d)
    return merged @ W_o.T
