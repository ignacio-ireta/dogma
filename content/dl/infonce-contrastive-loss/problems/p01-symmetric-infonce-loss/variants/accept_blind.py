"""Symmetric InfoNCE loss for N paired embeddings of dimension D."""

from __future__ import annotations

import math
import numbers

import torch


def symmetric_infonce_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return the symmetric InfoNCE loss as a 0-dim tensor with the dtype of a."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("a and b must be tensors")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be 2-D")
    if not torch.is_floating_point(a) or not torch.is_floating_point(b):
        raise ValueError("a and b must be floating-point")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    n, d = a.shape
    if n == 0 or d == 0:
        raise ValueError("N and D must be at least 1")
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        raise ValueError("temperature must be a real number")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")
    if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
        raise ValueError("embeddings must be finite")

    norm_a = a.norm(dim=1, keepdim=True)
    norm_b = b.norm(dim=1, keepdim=True)
    if bool((norm_a == 0).any()) or bool((norm_b == 0).any()):
        raise ValueError("a row has zero norm and no direction")
    a_hat = a / norm_a
    b_hat = b / norm_b

    sim = (a_hat @ b_hat.T) / tau
    diag = torch.diagonal(sim)

    row_max = sim.max(dim=1, keepdim=True).values
    lse_rows = row_max.squeeze(1) + torch.log(torch.exp(sim - row_max).sum(dim=1))
    col_max = sim.max(dim=0, keepdim=True).values
    lse_cols = col_max.squeeze(0) + torch.log(torch.exp(sim - col_max).sum(dim=0))

    loss_ab = (lse_rows - diag).mean()
    loss_ba = (lse_cols - diag).mean()
    return (loss_ab + loss_ba) / 2
