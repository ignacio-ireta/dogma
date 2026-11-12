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
    # Validation
    if not isinstance(reward_chosen, torch.Tensor) or not isinstance(reward_rejected, torch.Tensor):
        raise ValueError("Arguments must be torch.Tensors")

    if reward_chosen.ndim != 1 or reward_rejected.ndim != 1:
        raise ValueError("Tensors must be 1-D")

    if reward_chosen.shape != reward_rejected.shape:
        raise ValueError("Tensors must have the same shape")

    B = reward_chosen.shape[0]
    if B == 0:
        raise ValueError("Batch size B must be >= 1")

    if not reward_chosen.is_floating_point() or not reward_rejected.is_floating_point():
        raise ValueError("Tensors must have floating-point dtype")

    if reward_chosen.dtype != reward_rejected.dtype:
        raise ValueError("Tensors must have the same dtype")

    if not torch.all(torch.isfinite(reward_chosen)) or not torch.all(
        torch.isfinite(reward_rejected)
    ):
        raise ValueError("Tensors must contain only finite values")

    # Calculation
    # d = r_c - r_r
    # loss = -log(1 / (1 + exp(-d))) = log(1 + exp(-d))
    # To make this stable for large d (positive or negative) and maintain gradient at d=0:
    # log(1 + exp(-d)) = max(0, -d) + log(1 + exp(-|d|))
    # Using the constraint: max(x, 0) should be implemented as (x + abs(x)) / 2

    d = reward_chosen - reward_rejected

    # Stable form of log(1 + exp(-d))
    # Let x = -d
    x = -d
    # softplus(x) = log(1 + exp(x))
    # Stable softplus: max(0, x) + log(1 + exp(-abs(x)))
    # To ensure gradient at 0 is 1/2, we use (x + abs(x)) / 2 for max(0, x)

    max_0_x = (x + torch.abs(x)) / 2
    loss_per_pair = max_0_x + torch.log1p(torch.exp(-torch.abs(x)))

    loss = torch.mean(loss_per_pair)

    # Accuracy: fraction where r_c > r_r strictly
    correct = (reward_chosen > reward_rejected).to(reward_chosen.dtype)
    accuracy = torch.mean(correct)

    return loss, accuracy
