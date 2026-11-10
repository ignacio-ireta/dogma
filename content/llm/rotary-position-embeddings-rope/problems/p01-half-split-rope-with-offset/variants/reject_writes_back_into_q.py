from __future__ import annotations

import math

import torch


def _check(q: torch.Tensor, k: torch.Tensor, offset: int) -> None:
    """Validate the shapes, dtypes and offset of the contract."""
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise ValueError("q and k must be tensors")
    if q.ndim != 4 or k.ndim != 4 or not q.is_floating_point() or not k.is_floating_point():
        raise ValueError("q and k must be floating-point tensors of shape (B, H, T, d)")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    if q.dtype != k.dtype:
        raise ValueError("q and k must have the same dtype")
    d = q.shape[-1]
    if d == 0 or d % 2:
        raise ValueError("feature width must be even and at least 2")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative int")


def _half_split_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Wrong on purpose: writes the rotated second half back into the input."""
    half = x.shape[-1] // 2
    # Half-split pairing: feature i and feature i + d/2 form one complex number a + i b.
    a = x[..., :half]
    b = x[..., half:]
    # Multiply a + ib by exp(i phi) = cos phi + i sin phi:
    #   real part a cos - b sin, imaginary part a sin + b cos.
    rot_a = a * cos - b * sin
    rot_b = a * sin + b * cos
    x[..., half:].copy_(rot_b)
    return torch.cat([rot_a, rot_b], dim=-1)


def apply_rope_qk(
    q: torch.Tensor, k: torch.Tensor, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE to queries and keys, not values.

    Shapes: q, k (B, H, T, d). Token t sits at absolute position offset + t, so a cached
    decoder can rotate only the new tokens. Returns (q_rot, k_rot), each (B, H, T, d).
    """
    _check(q, k, offset)
    tokens, d = q.shape[-2], q.shape[-1]
    half = d // 2

    # theta_i = 10000 ** (-2 i / d) for each pair i. Pair 0 is the fastest.
    i = torch.arange(half, dtype=torch.float64)
    theta = torch.exp(-math.log(10000.0) * 2.0 * i / d)  # (d/2,)

    # Absolute positions start at offset, not at 0. Shape (T, 1).
    positions = torch.arange(offset, offset + tokens, dtype=torch.float64).unsqueeze(-1)

    # angle[t, i] = (offset + t) * theta_i, shape (T, d/2). Cast to the input dtype.
    angle = positions * theta
    cos = torch.cos(angle).to(q.dtype)
    sin = torch.sin(angle).to(q.dtype)

    # The same rotation applies to queries and keys. Values are never passed here.
    return _half_split_rotate(q, cos, sin), _half_split_rotate(k, cos, sin)
