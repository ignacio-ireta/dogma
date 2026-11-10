from __future__ import annotations

import math

import torch


def group_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(rewards, torch.Tensor):
        raise ValueError("rewards must be a tensor")
    if rewards.dim() != 2:
        raise ValueError("rewards must be 2-D with shape (P, G)")
    P, G = rewards.shape
    if P == 0:
        raise ValueError("need at least one prompt")
    if G < 2:
        raise ValueError("each group needs at least two responses")
    if not torch.is_floating_point(rewards):
        raise ValueError("rewards must be floating-point")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps must be a number")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")

    mean = rewards.mean(dim=1, keepdim=True)
    centred = rewards - mean
    # Population variance: divide by G, not G - 1.
    var = (centred * centred).sum(dim=1, keepdim=True) / G
    std = torch.sqrt(var)
    return centred / (std + eps)
