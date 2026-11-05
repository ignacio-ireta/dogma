from __future__ import annotations

import torch


def apply_rope_scaled(
    q: torch.Tensor, k: torch.Tensor, offset: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE with linear position interpolation to queries and keys.

    q, k: (B, H, T, d) floating tensors of the same dtype, d even and at least 2.
    Token t sits at (offset + t) / scale. Pair i is (x_i, x_{i+d/2}), theta_i = 10000**(-2i/d).
    Returns (q_rot, k_rot), each (B, H, T, d). Raise ValueError for invalid inputs.
    Do not mutate q, k, offset or scale.
    """
    raise NotImplementedError
