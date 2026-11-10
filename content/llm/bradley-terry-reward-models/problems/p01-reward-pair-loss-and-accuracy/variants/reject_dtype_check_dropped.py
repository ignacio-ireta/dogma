from __future__ import annotations

import torch


def _validate(reward_chosen: object, reward_rejected: object) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if not isinstance(reward_chosen, torch.Tensor) or not isinstance(reward_rejected, torch.Tensor):
        raise ValueError("both rewards must be torch tensors")
    if reward_chosen.ndim != 1 or reward_rejected.ndim != 1:
        raise ValueError("rewards must be 1-D tensors with one entry per pair")
    if reward_chosen.shape != reward_rejected.shape:
        raise ValueError("chosen and rejected rewards must have the same shape")
    if reward_chosen.numel() == 0:
        raise ValueError("at least one pair is required")
    if not reward_chosen.is_floating_point() or not reward_rejected.is_floating_point():
        raise ValueError("rewards must be floating-point tensors")
    if not (torch.isfinite(reward_chosen).all() and torch.isfinite(reward_rejected).all()):
        raise ValueError("rewards must be finite: NaN or inf found")


def bradley_terry_loss_and_accuracy(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrong on purpose: the check that both rewards share one dtype is missing.

    Model: P(chosen beats rejected) = sigmoid(d) with margin d = r_c - r_r, so the per-pair loss
    is -log(sigmoid(d)). Ties (d == 0) count as incorrect in the accuracy.
    """
    _validate(reward_chosen, reward_rejected)
    margin = reward_chosen - reward_rejected
    x = -margin
    per_pair = torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))
    loss = per_pair.mean()
    accuracy = (margin > 0).to(reward_chosen.dtype).mean()
    return loss, accuracy
