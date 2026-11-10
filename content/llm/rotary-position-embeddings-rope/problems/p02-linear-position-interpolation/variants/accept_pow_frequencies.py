from __future__ import annotations

import math

import torch


def _check(q: torch.Tensor, k: torch.Tensor, offset: int, scale: float) -> None:
    """Validate the tensors, the offset and the scale of the contract."""
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise ValueError("q and k must be tensors")
    if q.ndim != 4 or k.ndim != 4 or not q.is_floating_point() or not k.is_floating_point():
        raise ValueError("q and k must be floating-point tensors of shape (B, H, T, d)")
    if q.shape != k.shape or q.dtype != k.dtype:
        raise ValueError("q and k must have the same shape and dtype")
    d = q.shape[-1]
    if d == 0 or d % 2:
        raise ValueError("feature width must be even and at least 2")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative int")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale must be a finite number greater than 0")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite number greater than 0")


def _half_split_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x_i, x_{i+d/2}) by the angles behind cos and sin, each (T, d/2)."""
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]
    # (a + ib) * (cos + i sin) = (a cos - b sin) + i (a sin + b cos)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def apply_rope_scaled(
    q: torch.Tensor, k: torch.Tensor, offset: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE with linear position interpolation to queries and keys.

    Token t sits at the scaled position (offset + t) / scale. The base 10000 is unchanged,
    so pair 0 (theta_0 = 1) rotates by exactly (offset + t) / scale radians.
    """
    _check(q, k, offset, scale)
    tokens, d = q.shape[-2], q.shape[-1]

    # theta_i = 10000 ** (-2 i / d), one per pair.
    i = torch.arange(d // 2, dtype=torch.float64)
    theta = torch.pow(torch.tensor(10000.0, dtype=torch.float64), -2.0 * i / d)  # (d/2,)

    # Scale the whole position: the offset is divided together with t. Shape (T, 1).
    positions = (
        torch.arange(offset, offset + tokens, dtype=torch.float64) / float(scale)
    ).unsqueeze(-1)

    # angle[t, i] = ((offset + t) / scale) * theta_i, shape (T, d/2).
    angle = positions * theta
    cos = torch.cos(angle).to(q.dtype)
    sin = torch.sin(angle).to(q.dtype)
    return _half_split_rotate(q, cos, sin), _half_split_rotate(k, cos, sin)
