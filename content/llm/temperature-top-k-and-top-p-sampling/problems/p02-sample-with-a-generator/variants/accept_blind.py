from __future__ import annotations

import torch


def sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    if (
        not isinstance(probs, torch.Tensor)
        or probs.dtype != torch.float64
        or probs.dim() != 1
        or probs.shape[0] < 1
    ):
        raise ValueError("probs must be a non-empty 1-D float64 tensor")
    if not bool(torch.isfinite(probs).all()):
        raise ValueError("probs must be finite")
    if bool((probs < 0).any()):
        raise ValueError("probs must be non-negative")
    if abs(float(probs.sum()) - 1.0) > 1e-9:
        raise ValueError("probs must sum to 1 within 1e-9")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")

    u = torch.rand(1, dtype=torch.float64, generator=generator)
    cdf = torch.cumsum(probs, dim=0)
    idx = int(torch.searchsorted(cdf, u, right=True)[0])
    if idx >= probs.shape[0]:
        nonzero = torch.nonzero(probs > 0).flatten()
        idx = int(nonzero[-1])
    return idx
