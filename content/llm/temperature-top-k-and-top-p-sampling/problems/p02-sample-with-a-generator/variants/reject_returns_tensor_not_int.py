from __future__ import annotations

import torch


def sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Wrong on purpose: inverse-CDF sampling with one mistake, see the edit."""
    if not isinstance(probs, torch.Tensor) or probs.ndim != 1 or probs.shape[0] < 1:
        raise ValueError("probs must be a non-empty 1-D tensor")
    if probs.dtype != torch.float64:
        raise ValueError("probs must be float64")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator instance")
    if not bool(torch.isfinite(probs).all()) or bool((probs < 0).any()):
        raise ValueError("probs must be finite and non-negative")
    if abs(float(probs.sum().item()) - 1.0) > 1e-9:
        raise ValueError("probs must sum to 1 within 1e-9")

    V = probs.shape[0]
    # Step 1, cumulative distribution. cdf[i] is the mass of tokens 0..i. This is the function
    # the draw inverts. Compute it into a new tensor, so probs is never written.
    cdf = torch.cumsum(probs, dim=0)

    # Step 2, one uniform value in [0, 1), drawn from the passed generator. Passing generator=
    # keeps the global random state untouched and makes the result reproducible.
    u = torch.rand(1, generator=generator, dtype=torch.float64)

    # Step 3, invert the CDF. searchsorted(right=True) returns the first index with cdf > u.
    # A zero-probability index repeats its predecessor's cdf, so it can never be that first index.
    index = int(torch.searchsorted(cdf, u, right=True).item())
    if index < V:
        return torch.tensor(index)

    # Step 4, rounding guard. If the cdf total came out a hair below u, return the last token
    # that has mass. Clamping to V - 1 would be wrong, because that token may have probability 0.
    positive = torch.nonzero(probs > 0)
    return int(positive[-1].item())
