from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Return x with rotary position embedding applied along its sequence axis."""
    if x.dim() < 2:
        raise ValueError("x must have at least 2 dimensions")
    d = x.shape[-1]
    if d == 0 or d % 2 != 0:
        raise ValueError("feature width must be an even number of at least 2")

    L = x.shape[-2]
    half = d // 2  # noqa: F841
    t = torch.arange(L, dtype=torch.float64).unsqueeze(1)  # (L, 1)
    two_i = torch.arange(0, d, 2, dtype=torch.float64)  # 2i, shape (d/2,)
    theta = torch.pow(10000.0, -two_i / d)  # theta_i = 10000**(-2i/d)
    angle = t * theta  # (L, d/2)
    cos = torch.cos(angle).to(x.dtype)
    sin = torch.sin(angle).to(x.dtype)

    xe = x[..., 0::2]
    xo = x[..., 1::2]
    out_e = xe * cos - xo * sin
    out_o = xe * sin + xo * cos
    out = torch.stack([out_e, out_o], dim=-1)  # (..., L, half, 2)
    return out.reshape(x.shape)
