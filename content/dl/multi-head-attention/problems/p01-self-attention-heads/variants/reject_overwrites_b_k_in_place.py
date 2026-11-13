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
    """Multi-head self-attention, written out step by step (Vaswani et al. 2017, section 3.2.2).

    Shapes: X (B, L, d), projections (B, L, d), heads (B, h, L, d_k), scores (B, h, L, L),
    output (B, L, d), with d = d_model, h = num_heads and d_k = d / h.
    """
    # Validate the interface first. Each rule is a contract rule, so a silent broadcast cannot
    # hide a wrong shape.
    if X.ndim != 3:
        raise ValueError("X must be 3-D (B, L, d_model)")
    B, L, d = X.shape
    for name, W in (("W_q", W_q), ("W_k", W_k), ("W_v", W_v), ("W_o", W_o)):
        if tuple(W.shape) != (d, d):
            raise ValueError(f"{name} must have shape (d_model, d_model) in nn.Linear layout")
    for name, b in (("b_q", b_q), ("b_k", b_k), ("b_v", b_v), ("b_o", b_o)):
        if tuple(b.shape) != (d,):
            raise ValueError(f"{name} must have shape (d_model,)")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if d % num_heads:
        raise ValueError("num_heads must divide d_model")
    if mask is not None:
        if mask.dtype != torch.bool or tuple(mask.shape) != (L, L):
            raise ValueError("mask must be a torch.bool tensor of shape (L, L)")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every query row of the mask needs at least one allowed key")

    d_k = d // num_heads

    # Step 1, projections. For every token at once: y = x @ W.T + b, shape (B, L, d).
    # nn.Linear stores W as (out, in), so W.T maps input features to output features.
    Q = X @ W_q.T + b_q
    K = X @ W_k.T + b_k
    V = X @ W_v.T + b_v

    # Step 2, split heads. (B, L, d) -> (B, L, h, d_k) -> (B, h, L, d_k).
    # reshape cuts the feature axis into h contiguous blocks of d_k. transpose(1, 2) then puts
    # the head axis before the token axis, so each (L, d_k) matrix is one head. Without the
    # transpose, the token and head axes would mix.
    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, L, num_heads, d_k).transpose(1, 2)

    Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

    # Step 3, scores per head: Qh @ Kh^T / sqrt(d_k), shape (B, h, L, L). The division keeps the
    # dot products from growing with d_k, so the softmax does not saturate.
    scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(d_k)

    # Masked pairs get -inf, so softmax gives them exactly zero weight. Every row has an allowed
    # key (checked above), so no row is all -inf.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 4, softmax over the key axis, then the weighted sum of values per head.
    weights = torch.softmax(scores, dim=-1)  # (B, h, L, L)
    context = weights @ Vh  # (B, h, L, d_k)

    # Step 5, merge heads. Transpose back to (B, L, h, d_k), then flatten the head and feature
    # axes into d columns: head j's features land in block [j*d_k, (j+1)*d_k).
    merged = context.transpose(1, 2).reshape(B, L, d)

    # Step 6, output projection. The bias is one vector per output feature, added to every token.
    return merged @ W_o.T + b_o


_correct = multi_head_self_attention


def _change_in_place(v):
    if str(v.dtype) == "torch.bool":
        v.data.logical_not_()
    else:
        v.data.add_(1)


def multi_head_self_attention(*args, **kwargs):
    """Wrong on purpose: the result is right, but this overwrites `b_k` in place."""
    result = _correct(*args, **kwargs)
    target = args[6] if len(args) > 6 else kwargs.get("b_k")
    if type(target).__name__ in ("Tensor", "Parameter"):
        _change_in_place(target)
    return result


multi_head_self_attention.__wrapped__ = _correct
