from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

# LayerNorm epsilon from the contract. It sits inside the square root, before the division.
EPS = 1e-5

# The 16 keys of one pre-norm block, the same as dl/transformer-block p01.
BLOCK_KEYS = (
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
REQUIRED_KEYS = frozenset({"tok_emb", "pos_emb", "blocks", "ln_f_gamma", "ln_f_beta"})
ALLOWED_KEYS = REQUIRED_KEYS | {"lm_head"}


def _check_block(block: Any, d_model: int, dtype: torch.dtype) -> None:
    """Validate one block mapping: exactly the 16 keys, the shapes from its W_1, one dtype."""
    if not isinstance(block, Mapping) or set(block) != set(BLOCK_KEYS):
        raise ValueError("each block must be a mapping with exactly the 16 documented keys")
    if not all(torch.is_tensor(block[key]) for key in BLOCK_KEYS):
        raise ValueError("every block parameter must be a tensor")
    if block["W_1"].ndim != 2:
        raise ValueError("W_1 must be two-dimensional (D, F)")
    ff = int(block["W_1"].shape[1])
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
        if tuple(block[key].shape) != shape:
            raise ValueError(f"block parameter {key} must have shape {shape}")
        if block[key].dtype != dtype:
            raise ValueError("every parameter must have the same dtype as tok_emb")


def _check(idx: torch.Tensor, params: Mapping[str, Any], num_heads: int) -> bool:
    """Validate every rule in the contract. Returns True when the head is tied."""
    if not torch.is_tensor(idx) or idx.ndim != 2 or idx.dtype != torch.int64:
        raise ValueError("idx must be an int64 tensor of shape (B, T)")
    if idx.shape[0] == 0 or idx.shape[1] == 0:
        raise ValueError("idx must have at least one row and one token")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    keys = set(params)
    if keys < REQUIRED_KEYS or not keys <= ALLOWED_KEYS:
        raise ValueError(
            "params must contain the documented keys; lm_head is the only optional one"
        )

    vocab, d_model = _check_tables(idx, params, num_heads)
    dtype = params["tok_emb"].dtype
    _check_norms_and_blocks(params, d_model, dtype)

    tied = "lm_head" not in params
    if not tied:
        head = params["lm_head"]
        if not torch.is_tensor(head) or tuple(head.shape) != (d_model, vocab):
            raise ValueError("lm_head must have shape (D, V) when it is present")
        if head.dtype != dtype:
            raise ValueError("every parameter must have the same dtype as tok_emb")
    return tied


def _check_tables(idx: torch.Tensor, params: Mapping[str, Any], num_heads: int) -> tuple[int, int]:
    """Check tok_emb and pos_emb against idx and num_heads. Returns (V, D)."""
    tok = params["tok_emb"]
    if not torch.is_tensor(tok) or tok.ndim != 2 or not tok.is_floating_point():
        raise ValueError("tok_emb must be a floating-point tensor of shape (V, D)")
    vocab, d_model = tok.shape
    if d_model % num_heads != 0:
        raise ValueError("num_heads must divide D so each head gets D / num_heads features")
    dtype = tok.dtype

    pos = params["pos_emb"]
    if not torch.is_tensor(pos) or pos.ndim != 2 or pos.shape[1] != d_model:
        raise ValueError("pos_emb must have shape (C, D) with the same D as tok_emb")
    if pos.dtype != dtype:
        raise ValueError("every parameter must have the same dtype as tok_emb")
    if idx.shape[1] > pos.shape[0]:
        raise ValueError("sequence longer than the position table: T must be at most C")
    if int(idx.min()) < 0 or int(idx.max()) >= vocab:
        raise ValueError("token id outside the vocabulary: ids must lie in [0, V)")
    return vocab, d_model


def _check_norms_and_blocks(params: Mapping[str, Any], d_model: int, dtype: torch.dtype) -> None:
    """Check the final LayerNorm and every block against D and the shared dtype."""
    for name in ("ln_f_gamma", "ln_f_beta"):
        value = params[name]
        if not torch.is_tensor(value) or tuple(value.shape) != (d_model,):
            raise ValueError(f"{name} must have shape (D,)")
        if value.dtype != dtype:
            raise ValueError("every parameter must have the same dtype as tok_emb")

    blocks = params["blocks"]
    if not isinstance(blocks, (list, tuple)):
        raise ValueError("blocks must be a list of block mappings")
    for block in blocks:
        _check_block(block, d_model, dtype)


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
    """Multi-head causal attention. Shapes (B, T, D) in and out."""
    batch, tokens, d_model = q.shape
    d_head = d_model // num_heads

    # (B, T, D) -> (B, H, T, d_head): each head reads its own contiguous slice of features.
    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch, tokens, num_heads, d_head).transpose(1, 2)

    qh, kh, vh = split(q), split(k), split(v)
    # Scores scaled by 1/sqrt(d_head), the per-head width.
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, T, T)
    # Causal mask: query i may not look at key j > i. The diagonal stays visible.
    future = torch.ones(tokens, tokens, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # softmax over keys
    out = weights @ vh  # (B, H, T, d_head)
    return out.transpose(1, 2).reshape(batch, tokens, d_model)


def _pre_norm_block(x: torch.Tensor, p: Mapping[str, torch.Tensor], num_heads: int) -> torch.Tensor:
    """One pre-norm causal block: x + Attn(LN1(x)), then x' + FFN(LN2(x'))."""
    # Sub-layer 1: normalise the input, attend causally, add to the un-normalised stream.
    h = _layer_norm(x, p["ln1_gamma"], p["ln1_beta"])
    q = h @ p["W_q"] + p["b_q"]
    k = h @ p["W_k"] + p["b_k"]
    v = h @ p["W_v"] + p["b_v"]
    attn = _causal_attention(q, k, v, num_heads) @ p["W_o"] + p["b_o"]
    x_mid = x + attn
    # Sub-layer 2: normalise, expand to F with GELU, project back to D, add the residual.
    h2 = _layer_norm(x_mid, p["ln2_gamma"], p["ln2_beta"])
    hidden = _gelu(h2 @ p["W_1"] + p["b_1"])
    return x_mid + hidden @ p["W_2"] + p["b_2"]


def gpt_forward(idx: torch.Tensor, params: Mapping[str, Any], num_heads: int) -> torch.Tensor:
    """Return next-token logits (B, T, V) for a decoder-only GPT.

    Shapes: idx (B, T) int64; tok_emb (V, D); pos_emb (C, D); lm_head (D, V) when untied.
    Returns (B, T, V) in tok_emb's dtype. Raises ValueError for invalid inputs.
    """
    tied = _check(idx, params, num_heads)
    tokens = idx.shape[1]

    # 1. Embed: a row lookup for each token id, plus the row for its position.
    #    (B, T) ids -> (B, T, D). The position rows are (T, D) and broadcast over the batch.
    x = params["tok_emb"][idx] + params["pos_emb"][:tokens]

    # 2. Blocks: the residual stream passes through each pre-norm block in order.
    for block in params["blocks"]:
        x = _pre_norm_block(x, block, num_heads)

    # 3. Final LayerNorm, so the head sees a normalised stream.
    h = _layer_norm(x, params["ln_f_gamma"], params["ln_f_beta"])

    # 4. LM head. Tied: reuse the token table transposed, (D, V). This saves V * D parameters.
    #    Untied: use the separate (D, V) matrix.
    head = params["tok_emb"].T if tied else params["lm_head"]
    return h @ head  # (B, T, D) @ (D, V) -> (B, T, V)
