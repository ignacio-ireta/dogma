from __future__ import annotations

import torch


def apply_rope_qk(
    q: torch.Tensor, k: torch.Tensor, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE to queries and keys, not values.

    q, k: (B, H, T, d) floating tensors of the same dtype, d even and at least 2.
    Token t sits at position offset + t. Pair i is (x_i, x_{i+d/2}) with theta_i = 10000**(-2i/d).
    Returns (q_rot, k_rot), each (B, H, T, d). Raise ValueError for invalid inputs.
    Do not mutate q, k or offset.
    """
    raise NotImplementedError
