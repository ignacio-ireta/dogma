from __future__ import annotations

import math
from collections.abc import Mapping

import torch

# LayerNorm epsilon from the contract. It sits inside the square root, before the division.
EPS = 1e-5

# Parameter names, in the order the lesson introduces them.
PARAM_KEYS = (
    "ln1_gamma",
    "ln1_beta",
    "W_q",
    "b_q",
    "W_k",
    "b_k",
    "W_v",
    "b_v",
    "W_o",
    "b_o",
    "ln2_gamma",
    "ln2_beta",
    "W_1",
    "b_1",
    "W_2",
    "b_2",
)


def _check(x: torch.Tensor, params: Mapping[str, torch.Tensor], num_heads: int) -> int:
    """Validate every shape and dtype rule. Returns the feed-forward width F."""
    if not torch.is_tensor(x) or not x.is_floating_point() or x.ndim != 3:
        raise ValueError("x must be a floating-point tensor of shape (B, T, D)")
    if 0 in x.shape:
        raise ValueError("x must have at least one batch row, one token and one feature")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    _, _, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("num_heads must divide D so each head gets D / num_heads features")
    if not isinstance(params, Mapping) or set(params) != set(PARAM_KEYS):
        raise ValueError("params must contain exactly the documented keys")
    if not all(torch.is_tensor(params[key]) for key in PARAM_KEYS):
        raise ValueError("every parameter must be a tensor")
    if params["W_1"].ndim != 2:
        raise ValueError("W_1 must be two-dimensional (D, F)")
    ff = int(params["W_1"].shape[1])
    if ff < 1:
        raise ValueError("feed-forward width F must be at least 1")
    expected = {
        "ln1_gamma": (d_model,),
        "ln1_beta": (d_model,),
        "W_q": (d_model, d_model),
        "b_q": (d_model,),
        "W_k": (d_model, d_model),
        "b_k": (d_model,),
        "W_v": (d_model, d_model),
        "b_v": (d_model,),
        "W_o": (d_model, d_model),
        "b_o": (d_model,),
        "ln2_gamma": (d_model,),
        "ln2_beta": (d_model,),
        "W_1": (d_model, ff),
        "b_1": (ff,),
        "W_2": (ff, d_model),
        "b_2": (d_model,),
    }
    for key, shape in expected.items():
        if tuple(params[key].shape) != shape:
            raise ValueError(f"parameter {key} must have shape {shape}")
        if params[key].dtype != x.dtype:
            raise ValueError("every parameter must have the same dtype as x")
    return ff


def _layer_norm(u: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the last axis, with biased variance (divide by D, not D - 1)."""
    mean = u.mean(dim=-1, keepdim=True)
    variance = ((u - mean) ** 2).mean(dim=-1, keepdim=True)
    return gamma * (u - mean) * torch.rsqrt(variance + EPS) + beta


def _gelu(z: torch.Tensor) -> torch.Tensor:
    """Exact GELU, z * Phi(z) with Phi the standard normal CDF written through erf."""
    return 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Multi-head scaled dot-product attention with a causal mask. Shapes (B, T, D) in and out."""
    batch, tokens, d_model = q.shape
    d_head = d_model // num_heads

    # Split the feature axis into heads: (B, T, D) -> (B, H, T, d_head). Each head sees its own
    # contiguous slice of features, so the head axis must come before the token axis.
    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch, tokens, num_heads, d_head).transpose(1, 2)

    qh, kh, vh = split(q), split(k), split(v)

    # Scores scaled by 1/sqrt(d_head), the per-head dimension, not by D.
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, T, T)

    # Causal mask: query position i may not look at key position j > i.
    future = torch.ones(tokens, tokens, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))

    # Softmax over keys (the last axis). Row i has at least one finite entry (j = i).
    weights = torch.softmax(scores, dim=-1)
    out = weights @ vh  # (B, H, T, d_head)

    # Merge heads back: (B, H, T, d_head) -> (B, T, D).
    return out.transpose(1, 2).reshape(batch, tokens, d_model)


def pre_norm_block(
    x: torch.Tensor, params: Mapping[str, torch.Tensor], num_heads: int
) -> torch.Tensor:
    """One pre-norm decoder block: x + MHA(LN(x)), then the same with a GELU feed-forward.

    Shapes: x (B, T, D); W_q, W_k, W_v, W_o (D, D); W_1 (D, F); W_2 (F, D).
    Projections use row-vector form y = x @ W + b. Returns (B, T, D) in x's dtype.
    """
    _check(x, params, num_heads)

    # Sub-layer 1: normalise the input, attend causally, add the result back to the stream.
    # The residual adds to the un-normalised x: the stream itself is never normalised.
    h = _layer_norm(x, params["ln1_gamma"], params["ln1_beta"])
    q = h @ params["W_q"] + params["b_q"]
    k = h @ params["W_k"] + params["b_k"]
    v = h @ params["W_v"] + params["b_v"]
    attn = _causal_attention(q, k, v, num_heads) @ params["W_o"] + params["b_o"]
    x_mid = x + attn

    # Sub-layer 2: normalise, expand to F with GELU, project back to D, add the residual.
    h2 = _layer_norm(x_mid, params["ln2_gamma"], params["ln2_beta"])
    hidden = _gelu(h2 @ params["W_1"] + params["b_1"])
    ffn = hidden @ params["W_2"] + params["b_2"]
    return x_mid + ffn
