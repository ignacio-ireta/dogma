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
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise ValueError("q and k must be tensors")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be 4-D tensors")
    if not q.is_floating_point() or not k.is_floating_point():
        raise ValueError("q and k must be floating-point tensors")
    if tuple(q.shape) != tuple(k.shape):
        raise ValueError("q and k must have the same shape")
    if q.dtype != k.dtype:
        raise ValueError("q and k must have the same dtype")
    d = int(q.shape[-1])
    if d % 2 != 0 or d < 2:
        raise ValueError("feature width must be even and at least 2")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an int")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    T = int(q.shape[2])
    half = d // 2
    dtype = q.dtype
    device = q.device

    # Angles are computed in float64 for accuracy, then cast back to the input dtype.
    i = torch.arange(half, dtype=torch.float64, device=device)
    theta = torch.pow(torch.tensor(10000.0, dtype=torch.float64, device=device), -2.0 * i / d)
    pos = torch.arange(offset, offset + T, dtype=torch.float64, device=device)
    phi = pos.unsqueeze(1) * theta.unsqueeze(0)  # (T, d/2)
    cos = torch.cos(phi).to(dtype)
    sin = torch.sin(phi).to(dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        a = x[..., :half]
        b = x[..., half:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

    return rotate(q), rotate(k)
