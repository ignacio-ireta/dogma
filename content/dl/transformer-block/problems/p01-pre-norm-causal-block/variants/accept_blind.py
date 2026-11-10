from __future__ import annotations

import math
from collections.abc import Mapping

import torch

_KEYS = {
    "ln1_gamma", "ln1_beta", "W_q", "b_q", "W_k", "b_k", "W_v", "b_v", "W_o", "b_o",
    "ln2_gamma", "ln2_beta", "W_1", "b_1", "W_2", "b_2",
}  # fmt: skip
_EPS = 1e-5


def _layer_norm(u: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mu = u.mean(dim=-1, keepdim=True)
    var = ((u - mu) ** 2).mean(dim=-1, keepdim=True)
    return gamma * (u - mu) / torch.sqrt(var + _EPS) + beta


def _gelu(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))


def pre_norm_block(
    x: torch.Tensor, params: Mapping[str, torch.Tensor], num_heads: int
) -> torch.Tensor:
    """One pre-norm decoder block: x + MHA(LN(x)), then the same with a GELU feed-forward.

    Shapes: x (B, T, D); W_q, W_k, W_v, W_o (D, D); W_1 (D, F); W_2 (F, D).
    Projections use row-vector form y = x @ W + b. Returns (B, T, D) in x's dtype.
    Causal: query i must not attend to key j > i. LayerNorm uses biased variance, eps 1e-5.
    GELU is the exact erf form. Raise ValueError for invalid shapes, keys or dtypes.
    Do not mutate x or params.
    """
    if not torch.is_tensor(x) or x.dim() != 3 or not x.is_floating_point():
        raise ValueError("x must be a 3-D floating-point tensor of shape (B, T, D)")
    if any(s == 0 for s in x.shape):
        raise ValueError("x must not have a zero-size axis")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be a positive int")
    B, T, D = x.shape
    if D % num_heads != 0:
        raise ValueError("num_heads must divide D so each head gets D / num_heads features")

    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    keys = set(params.keys())
    if keys != _KEYS:
        missing = sorted(_KEYS - keys)
        extra = sorted(keys - _KEYS)
        raise ValueError(
            f"params must have exactly the 16 required keys; missing={missing} extra={extra}"
        )
    for k, v in params.items():
        if not torch.is_tensor(v):
            raise ValueError(f"params[{k!r}] must be a tensor")

    F = params["W_1"].shape[1] if params["W_1"].dim() == 2 else -1
    expected = {
        "ln1_gamma": (D,), "ln1_beta": (D,),
        "W_q": (D, D), "b_q": (D,), "W_k": (D, D), "b_k": (D,),
        "W_v": (D, D), "b_v": (D,), "W_o": (D, D), "b_o": (D,),
        "ln2_gamma": (D,), "ln2_beta": (D,),
        "W_1": (D, F), "b_1": (F,), "W_2": (F, D), "b_2": (D,),
    }  # fmt: skip
    if F < 1:
        raise ValueError("W_1 must have shape (D, F) with F >= 1")
    for k, shape in expected.items():
        if tuple(params[k].shape) != shape:
            raise ValueError(f"params[{k!r}] must have shape {shape}, got {tuple(params[k].shape)}")
        if params[k].dtype != x.dtype:
            raise ValueError(f"params[{k!r}] dtype must match x")

    dh = D // num_heads
    # Attention sub-layer.
    h = _layer_norm(x, params["ln1_gamma"], params["ln1_beta"])
    q = h @ params["W_q"] + params["b_q"]
    k = h @ params["W_k"] + params["b_k"]
    v = h @ params["W_v"] + params["b_v"]

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, T, num_heads, dh).transpose(1, 2)  # (B, H, T, dh)

    q, k, v = split(q), split(k), split(v)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1) @ v  # (B, H, T, dh)
    attn = attn.transpose(1, 2).reshape(B, T, D)
    a = attn @ params["W_o"] + params["b_o"]
    x1 = x + a

    # Feed-forward sub-layer.
    h2 = _layer_norm(x1, params["ln2_gamma"], params["ln2_beta"])
    f = _gelu(h2 @ params["W_1"] + params["b_1"]) @ params["W_2"] + params["b_2"]
    return x1 + f
