from __future__ import annotations

import torch


def bradley_terry_loss_and_accuracy(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean Bradley-Terry loss, strict pair accuracy) for B reward pairs.

    The loss is the mean over pairs of -log(sigmoid(r_c - r_r)), computed in a form that stays
    finite for large margins in either direction. The accuracy is the fraction of pairs with
    r_c > r_r strictly; ties count as incorrect. Both results are 0-dimensional tensors with the
    input dtype. Raise ValueError for non-tensors, non-1-D or mismatched shapes, empty input,
    non-floating or mismatched dtypes, and non-finite values. Never mutate the inputs.
    """
    if not isinstance(reward_chosen, torch.Tensor) or not isinstance(reward_rejected, torch.Tensor):
        raise ValueError("both rewards must be torch.Tensor values")
    if reward_chosen.dim() != 1 or reward_rejected.dim() != 1:
        raise ValueError("both rewards must be one-dimensional")
    if reward_chosen.shape != reward_rejected.shape:
        raise ValueError("reward tensors must have the same shape")
    if reward_chosen.shape[0] == 0:
        raise ValueError("a batch needs at least one pair")
    if not torch.is_floating_point(reward_chosen) or not torch.is_floating_point(reward_rejected):
        raise ValueError("rewards must be floating point")
    if reward_chosen.dtype != reward_rejected.dtype:
        raise ValueError("reward tensors must share a dtype")
    if not bool(torch.isfinite(reward_chosen).all()) or not bool(
        torch.isfinite(reward_rejected).all()
    ):
        raise ValueError("rewards must be finite")

    margin = reward_chosen - reward_rejected
    # -log(sigmoid(d)) = softplus(-d) = max(-d, 0) + log(1 + exp(-|d|)), stable for any margin.
    per_pair = torch.clamp(-margin, min=0) + torch.log1p(torch.exp(-torch.abs(margin)))
    loss = per_pair.mean()
    accuracy = (reward_chosen > reward_rejected).to(reward_chosen.dtype).mean()
    return loss, accuracy
