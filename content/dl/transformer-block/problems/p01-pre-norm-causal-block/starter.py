from __future__ import annotations

from collections.abc import Mapping

import torch


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
    raise NotImplementedError
