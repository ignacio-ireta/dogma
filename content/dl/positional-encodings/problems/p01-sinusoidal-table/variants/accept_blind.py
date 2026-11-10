from __future__ import annotations

import torch


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return the (max_len, d_model) sinusoidal positional encoding table, float64."""
    if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 0:
        raise ValueError("max_len must be a non-negative integer")
    if isinstance(d_model, bool) or not isinstance(d_model, int):
        raise ValueError("d_model must be a positive even integer")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model must be a positive even integer")

    pos = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)  # (L, 1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float64)  # 2i, shape (d/2,)
    div = torch.pow(10000.0, two_i / d_model)  # 10000**(2i/d)
    angles = pos / div  # (L, d/2)

    out = torch.empty(max_len, d_model, dtype=torch.float64)
    out[:, 0::2] = torch.sin(angles)
    out[:, 1::2] = torch.cos(angles)
    return out
