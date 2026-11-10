from __future__ import annotations

import numbers

import torch


def _checked_inputs(
    logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, gamma: float, baseline: bool
) -> float:
    """Validate shapes, dtypes, ranges and scalars before any computation."""
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise ValueError("logits must be a floating-point tensor")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, A)")
    n_episodes, n_steps, n_actions = logits.shape
    if min(n_episodes, n_actions) < 1:
        raise ValueError("the batch, the episode length and the action count must be at least 1")
    if (
        not isinstance(actions, torch.Tensor)
        or actions.dtype in (torch.bool,)
        or (actions.dtype.is_floating_point)
    ):
        raise ValueError("actions must be an integer tensor")
    if tuple(actions.shape) != (n_episodes, n_steps):
        raise ValueError("actions must have shape (B, T)")
    if actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= n_actions):
        raise ValueError("action index out of range for the logits")
    if not isinstance(rewards, torch.Tensor) or not rewards.is_floating_point():
        raise ValueError("rewards must be a floating-point tensor")
    if tuple(rewards.shape) != (n_episodes, n_steps):
        raise ValueError("rewards must have shape (B, T)")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        raise ValueError("gamma must be a real number")
    value = float(gamma)
    if not 0.0 <= value <= 1.0:  # NaN fails the comparison too
        raise ValueError("gamma must lie in [0, 1]")
    if not isinstance(baseline, bool):
        raise ValueError("baseline must be a bool")
    return value


def reinforce_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    baseline: bool = False,
) -> torch.Tensor:
    """Wrong on purpose: accepts zero-length episodes and returns zero.

    logits: float (B, T, A); actions: integer (B, T) in [0, A); rewards: float (B, T), finite,
    rewards[b, t] follows the action at step t. A_bt is the reward-to-go G_bt with discount gamma,
    minus, if baseline is True, the batch mean of G over b at the same step t.
    Returns a 0-dim tensor with the dtype of logits. Raises ValueError on invalid input.
    Never mutates inputs and never calls torch.nn, torch.distributions or torch.autograd.
    """
    discount = _checked_inputs(logits, actions, rewards, gamma, baseline)
    n_episodes, n_steps, _ = logits.shape

    # Reward-to-go by a backward pass: G_t = r_{t+1} + gamma * G_{t+1}, with nothing after step T-1.
    # The returns are data, so they do not carry gradient to the logits.
    returns = torch.zeros((n_episodes, n_steps), dtype=logits.dtype)
    running = torch.zeros(n_episodes, dtype=logits.dtype)
    for t in range(n_steps - 1, -1, -1):
        running = rewards[:, t].to(logits.dtype) + discount * running
        returns[:, t] = running

    advantage = returns
    if baseline:
        # Subtract the batch mean at each step. It is the same for every episode at that step, so
        # it does not favour any one trajectory. (It does depend on the batch; see the lesson.)
        advantage = returns - returns.mean(dim=0, keepdim=True)

    # log pi(a | s) from log-softmax: subtracting the maximum first keeps extreme logits finite.
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # The minus sign turns the ascent direction into a loss to minimise. Dividing by B keeps the
    # scale independent of the batch size.
    return -(chosen * advantage).sum() / n_episodes
