from __future__ import annotations

import math

import torch


def dpo_loss_and_implicit_rewards(  # noqa: PLR0912
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    named = {
        "policy_chosen": policy_chosen,
        "policy_rejected": policy_rejected,
        "ref_chosen": ref_chosen,
        "ref_rejected": ref_rejected,
    }
    for name, value in named.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if value.dim() != 1:
            raise ValueError(f"{name} must be 1-D")
    shape = policy_chosen.shape
    for name, value in named.items():
        if value.shape != shape:
            raise ValueError(f"{name} shape differs from policy_chosen")
    if shape[0] == 0:
        raise ValueError("batch must be non-empty")
    dtype = policy_chosen.dtype
    if not torch.is_floating_point(policy_chosen):
        raise ValueError("log-probabilities must be floating-point")
    for name, value in named.items():
        if value.dtype != dtype:
            raise ValueError(f"{name} dtype differs from policy_chosen")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        if bool((value > 0).any()):
            raise ValueError(f"{name} must be log-probabilities (<= 0), not probabilities")

    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be a number")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and > 0")

    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)

    z = chosen_reward - rejected_reward
    # -log(sigmoid(z)) = softplus(-z), in a form stable for large |z|.
    per_pair = torch.clamp(-z, min=0) + torch.log1p(torch.exp(-torch.abs(z)))
    loss = per_pair.mean()
    return loss, chosen_reward, rejected_reward
