from __future__ import annotations

import torch


def dpo_loss_and_implicit_rewards(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (DPO loss, chosen implicit reward, rejected implicit reward) for B pairs.

    Implicit reward: beta * (policy log-prob - reference log-prob) for each response. Loss: mean
    over pairs of -log(sigmoid(r_chosen - r_rejected)), computed in a form stable for large
    margins. All four log-prob inputs are (B,) floating tensors with the same dtype, finite and
    <= 0. beta must be finite and > 0. Raise ValueError for any violation, and never mutate inputs.
    """
    raise NotImplementedError
