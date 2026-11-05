from __future__ import annotations

import torch


def reinforce_loss(
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
    raise NotImplementedError
