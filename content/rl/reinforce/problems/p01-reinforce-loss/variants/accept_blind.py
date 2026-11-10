from __future__ import annotations

import math
import numbers

import torch


def reinforce_loss(  # noqa: PLR0912
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    baseline: bool = False,
) -> torch.Tensor:
    """Return the REINFORCE loss -(1/B) sum_{b,t} log pi(a_bt | logits_bt) * A_bt.

    logits: float (B, T, A); actions: integer (B, T) in [0, A); rewards: float (B, T), finite,
    rewards[b, t] follows the action at step t. A_bt is the reward-to-go G_bt with discount gamma,
    minus, if baseline is True, the batch mean of G over b at the same step t.
    Returns a 0-dim tensor with the dtype of logits. Raises ValueError on invalid input.
    Never mutates inputs and never calls torch.nn, torch.distributions or torch.autograd.
    """
    for name, x in (("logits", logits), ("actions", actions), ("rewards", rewards)):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")

    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if torch.is_complex(logits):
        raise ValueError("logits must be real")
    if actions.dtype in (torch.bool,) or actions.is_floating_point() or actions.is_complex():
        raise ValueError("actions must be an integer tensor")
    if not torch.is_floating_point(rewards):
        raise ValueError("rewards must be floating point")

    if logits.dim() != 3:
        raise ValueError("logits must have shape (B, T, A)")
    B, T, A = logits.shape
    if B < 1 or T < 1 or A < 1:
        raise ValueError("batch, episode length and action count must all be at least 1")
    if actions.shape != (B, T):
        raise ValueError("actions must have shape (B, T)")
    if rewards.shape != (B, T):
        raise ValueError("rewards must have shape (B, T)")
    if actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= A):
        raise ValueError("actions must lie in [0, A)")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")

    if isinstance(gamma, (bool,)) or not isinstance(gamma, numbers.Real):
        raise ValueError("gamma must be a real number")
    g = float(gamma)
    if math.isnan(g) or not (0.0 <= g <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")
    if not isinstance(baseline, bool):
        raise ValueError("baseline must be a bool")

    rew = rewards.to(logits.dtype)
    returns = [None] * T
    running = torch.zeros(B, dtype=logits.dtype, device=logits.device)
    for t in range(T - 1, -1, -1):
        running = rew[:, t] + g * running
        returns[t] = running
    G = torch.stack(returns, dim=1)

    if baseline:
        adv = G - G.mean(dim=0, keepdim=True)
    else:
        adv = G

    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, actions.unsqueeze(-1).long()).squeeze(-1)
    return -(chosen * adv).sum() / B
