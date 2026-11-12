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
    if reward_chosen.dtype != reward_rejected.dtype:
        raise ValueError("chosen and rejected rewards must share one dtype")
    if not (torch.isfinite(reward_chosen).all() and torch.isfinite(reward_rejected).all()):
        raise ValueError("rewards must be finite: NaN or inf found")


def bradley_terry_loss_and_accuracy(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean Bradley-Terry loss, strict pair accuracy) for B reward pairs.

    Model: P(chosen beats rejected) = sigmoid(d) with margin d = r_c - r_r, so the per-pair loss
    is -log(sigmoid(d)). Ties (d == 0) count as incorrect in the accuracy.
    """
    _validate(reward_chosen, reward_rejected)
    # Step 1: the margin. Only this difference enters the model, so adding one constant to both
    # rewards changes nothing.
    margin = reward_chosen - reward_rejected
    # Step 2: -log(sigmoid(d)) = softplus(-d). Use the identity
    #   softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    # which never exponentiates a large positive number. The naive log(sigmoid(d)) underflows to
    # log(0) = -inf for large negative margins.
    # max(x, 0) is written as (x + |x|) / 2: the same values, and the right gradient at a tie.
    # At x = 0 autograd gives abs a slope of 0, so this form has slope 1/2 there, which is the
    # slope of softplus at 0. clamp(x, min=0) has a kink at 0 and gives a tied pair no gradient.
    x = -margin
    magnitude = torch.abs(x)
    per_pair = 0.5 * (x + magnitude) + torch.log1p(torch.exp(-magnitude))
    # Step 3: the loss is the mean over pairs. Gradients flow through both reward tensors.
    loss = per_pair.mean()
    # Step 4: a pair is correct only when the chosen reward is strictly larger.
    accuracy = (margin > 0).to(reward_chosen.dtype).mean()
    return loss, accuracy
