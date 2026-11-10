from __future__ import annotations

import math

import torch


def _validate(rewards: object, eps: object) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 2:
        raise ValueError("rewards must be a 2-D tensor of shape (P, G)")
    prompts, group_size = rewards.shape
    if prompts == 0:
        raise ValueError("at least one prompt is required")
    if group_size < 2:
        # A single sample has no group to compare with, and no spread to normalise by.
        raise ValueError("each group needs at least two sampled responses")
    if not rewards.is_floating_point():
        raise ValueError("rewards must be floating-point")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite: NaN or inf found")
    if isinstance(eps, bool) or not isinstance(eps, float | int):
        raise ValueError("eps must be a real number")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and strictly positive")


def group_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Wrong on purpose: the input rewards are centred in place, so the caller's tensor changes"""
    _validate(rewards, eps)
    group_size = rewards.shape[1]
    # Step 1: the group mean, one per prompt. Shape (P, 1) so it broadcasts over the group.
    mean = rewards.mean(dim=1, keepdim=True)
    rewards -= mean
    centred = rewards
    # Step 2: the population variance divides by G, not G - 1. That is the convention here.
    variance = (centred * centred).sum(dim=1, keepdim=True) / group_size
    std = torch.sqrt(variance)
    # Step 3: eps is added outside the square root. An all-equal group has std 0, and the
    # centred values are 0, so the advantage is 0 / eps = 0 instead of 0 / 0 = NaN.
    return centred / (std + eps)
