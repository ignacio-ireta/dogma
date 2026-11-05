from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def gpt_forward(idx: torch.Tensor, params: Mapping[str, Any], num_heads: int) -> torch.Tensor:
    """Return next-token logits (B, T, V) for a decoder-only GPT.

    idx (B, T) int64 token ids; tok_emb (V, D); pos_emb (C, D); blocks: a list of 16-key
    pre-norm block mappings (see dl/transformer-block); ln_f_gamma and ln_f_beta (D,); lm_head
    (D, V) present means untied, absent means tied to tok_emb. Embedding = token + position,
    then the blocks, final LayerNorm (biased variance, eps 1e-5), exact GELU inside the blocks,
    causal attention, and the head. Raise ValueError for invalid inputs. Do not mutate inputs.
    """
    raise NotImplementedError
