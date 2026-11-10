from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

_EPS = 1e-5
_BLOCK_KEYS = {
    "ln1_gamma", "ln1_beta", "W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o",
    "ln2_gamma", "ln2_beta", "W_1", "b_1", "W_2", "b_2",
}  # fmt: skip
_REQUIRED_KEYS = {"tok_emb", "pos_emb", "blocks", "ln_f_gamma", "ln_f_beta"}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"lm_head"}


def _layer_norm(u: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mu = u.mean(dim=-1, keepdim=True)
    var = ((u - mu) ** 2).mean(dim=-1, keepdim=True)
    return gamma * (u - mu) / torch.sqrt(var + _EPS) + beta


def _gelu(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _check_block(block: Any, d: int, dtype: torch.dtype, index: int) -> int:
    if not isinstance(block, Mapping):
        raise ValueError(f"block {index} must be a mapping")
    keys = set(block.keys())
    if keys != _BLOCK_KEYS:
        missing = sorted(_BLOCK_KEYS - keys)
        extra = sorted(keys - _BLOCK_KEYS)
        raise ValueError(f"block {index} has wrong keys: missing {missing}, extra {extra}")
    for name in _BLOCK_KEYS:
        if not torch.is_tensor(block[name]):
            raise ValueError(f"block {index} entry {name} must be a tensor")
        if block[name].dtype != dtype:
            raise ValueError(f"block {index} entry {name} dtype differs from tok_emb")
    expected = {
        "ln1_gamma": (d,), "ln1_beta": (d,), "ln2_gamma": (d,), "ln2_beta": (d,),
        "W_q": (d, d), "W_k": (d, d), "W_v": (d, d), "W_o": (d, d),
        "b_q": (d,), "b_k": (d,), "b_v": (d,), "b_o": (d,),
        "b_2": (d,),
    }  # fmt: skip
    for name, shape in expected.items():
        if tuple(block[name].shape) != shape:
            raise ValueError(f"block {index} entry {name} has wrong shape")
    w1 = block["W_1"]
    if w1.dim() != 2 or w1.shape[0] != d or w1.shape[1] < 1:
        raise ValueError(f"block {index} W_1 must have shape (D, F) with F >= 1")
    f = int(w1.shape[1])
    if tuple(block["b_1"].shape) != (f,):
        raise ValueError(f"block {index} b_1 has wrong shape")
    if tuple(block["W_2"].shape) != (f, d):
        raise ValueError(f"block {index} W_2 has wrong shape")
    return f


def _block_forward(
    x: torch.Tensor, blk: Mapping[str, torch.Tensor], num_heads: int
) -> torch.Tensor:
    B, T, D = x.shape
    dh = D // num_heads
    h = _layer_norm(x, blk["ln1_gamma"], blk["ln1_beta"])
    q = h @ blk["W_q"] + blk["b_q"]
    k = h @ blk["W_k"] + blk["b_k"]
    v = h @ blk["W_v"] + blk["b_v"]
    q = q.reshape(B, T, num_heads, dh).transpose(1, 2)
    k = k.reshape(B, T, num_heads, dh).transpose(1, 2)
    v = v.reshape(B, T, num_heads, dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    att = torch.softmax(scores, dim=-1)
    out = (att @ v).transpose(1, 2).reshape(B, T, D)
    x = x + (out @ blk["W_o"] + blk["b_o"])
    h2 = _layer_norm(x, blk["ln2_gamma"], blk["ln2_beta"])
    m = _gelu(h2 @ blk["W_1"] + blk["b_1"]) @ blk["W_2"] + blk["b_2"]
    return x + m


def gpt_forward(idx: torch.Tensor, params: Mapping[str, Any], num_heads: int) -> torch.Tensor:  # noqa: PLR0912, PLR0915
    """Return next-token logits (B, T, V) for a decoder-only GPT.

    idx (B, T) int64 token ids; tok_emb (V, D); pos_emb (C, D); blocks: a list of 16-key
    pre-norm block mappings (see dl/transformer-block); ln_f_gamma and ln_f_beta (D,); lm_head
    (D, V) present means untied, absent means tied to tok_emb. Embedding = token + position,
    then the blocks, final LayerNorm (biased variance, eps 1e-5), exact GELU inside the blocks,
    causal attention, and the head. Raise ValueError for invalid inputs. Do not mutate inputs.
    """
    # idx checks
    if not torch.is_tensor(idx) or idx.dim() != 2:
        raise ValueError("idx must be a 2-D tensor")
    if idx.dtype != torch.int64:
        raise ValueError("idx must be int64")
    if idx.shape[0] == 0 or idx.shape[1] == 0:
        raise ValueError("idx must not have a zero-size axis")

    # num_heads
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive int")

    # params structure
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    keys = set(params.keys())
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"params is missing keys {sorted(missing)}")
    extra = keys - _ALLOWED_KEYS
    if extra:
        raise ValueError(f"params has unexpected keys {sorted(extra)}")
    for name in keys:
        if name == "blocks":
            continue
        if not torch.is_tensor(params[name]):
            raise ValueError(f"params entry {name} must be a tensor")
    if not isinstance(params["blocks"], list):
        raise ValueError("blocks must be a list")

    tok_emb = params["tok_emb"]
    if not tok_emb.is_floating_point() or tok_emb.dim() != 2:
        raise ValueError("tok_emb must be a floating 2-D tensor")
    dtype = tok_emb.dtype
    V, D = int(tok_emb.shape[0]), int(tok_emb.shape[1])
    if D % num_heads != 0:
        raise ValueError("num_heads must divide D")

    pos_emb = params["pos_emb"]
    if pos_emb.dim() != 2 or pos_emb.shape[1] != D:
        raise ValueError("pos_emb must have shape (C, D)")
    C = int(pos_emb.shape[0])

    for name in ("ln_f_gamma", "ln_f_beta"):
        if tuple(params[name].shape) != (D,):
            raise ValueError(f"{name} must have shape (D,)")
    if "lm_head" in params and tuple(params["lm_head"].shape) != (D, V):
        raise ValueError("lm_head must have shape (D, V)")

    for name in ("pos_emb", "ln_f_gamma", "ln_f_beta", "lm_head"):
        if name in params and params[name].dtype != dtype:
            raise ValueError(f"{name} dtype differs from tok_emb")

    for i, blk in enumerate(params["blocks"]):
        _check_block(blk, D, dtype, i)

    B, T = int(idx.shape[0]), int(idx.shape[1])  # noqa: F841
    if T > C:
        raise ValueError(f"sequence length {T} exceeds context {C}")
    if bool(((idx < 0) | (idx >= V)).any()):
        raise ValueError("token id outside the vocabulary")

    # forward
    x = tok_emb[idx] + pos_emb[:T].unsqueeze(0)
    for blk in params["blocks"]:
        x = _block_forward(x, blk, num_heads)
    h = _layer_norm(x, params["ln_f_gamma"], params["ln_f_beta"])
    head = params["lm_head"] if "lm_head" in params else tok_emb.transpose(0, 1)
    return h @ head
