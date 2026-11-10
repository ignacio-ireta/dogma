from __future__ import annotations

import math

import torch


def _validate(
    tensors: tuple[object, ...],
    beta: object,
) -> None:
    """Reject inputs outside the contract before any arithmetic runs."""
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("all four log-probability arguments must be torch tensors")
    first = tensors[0]
    assert isinstance(first, torch.Tensor)  # narrowed by the check above
    if first.ndim != 1:
        raise ValueError("log-probabilities must be 1-D tensors with one entry per pair")
    if first.numel() == 0:
        raise ValueError("at least one pair is required")
    for t in tensors:
        assert isinstance(t, torch.Tensor)
        if t.shape != first.shape:
            raise ValueError("all four log-probability tensors must have the same shape")
        if not t.is_floating_point():
            raise ValueError("log-probabilities must be floating-point tensors")
        if t.dtype != first.dtype:
            raise ValueError("all four log-probability tensors must share one dtype")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("log-probabilities must be finite: NaN or inf found")
        if bool((t > 0).any()):
            # A log-probability of a response is at most 0. A positive value is usually a
            # probability passed where a log-probability was expected.
            raise ValueError(
                "values above zero are not log-probabilities: did you pass probabilities?"
            )
    if not isinstance(beta, float | int) or isinstance(beta, bool):
        raise ValueError("beta must be a real number")
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("beta must be finite and strictly positive")


def dpo_loss_and_implicit_rewards(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrong on purpose: return (DPO loss, chosen implicit reward, rejected implicit reward) for B pairs."""  # noqa: E501
    _validate((policy_chosen, policy_rejected, ref_chosen, ref_rejected), beta)
    # Step 1: implicit rewards. Each is beta times the log-ratio of policy to reference for one
    # response. The log-ratio is a difference of log-probs, so an unknown normaliser that depends
    # only on the prompt cancels inside each pair.
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    # Step 2: the margin the Bradley-Terry model sees.
    margin = chosen_reward - rejected_reward
    # Step 3: -log(sigmoid(m)) = softplus(-m) = max(-m, 0) + log(1 + exp(-|m|)). The split form
    # stays finite for margins of either sign and large magnitude.
    x = -margin
    per_pair = torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))
    # Step 4: average over pairs. Gradients reach the policy arguments through the rewards.
    loss = per_pair.mean()
    return loss, chosen_reward, rejected_reward
