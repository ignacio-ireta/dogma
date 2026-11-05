from __future__ import annotations

import torch


def grpo_token_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> torch.Tensor:
    """Return the scalar GRPO loss over (G, T) sampled tokens with per-response masks.

    Per token: rho = exp(logp_new - logp_old); surrogate = min(rho * A, clip(rho, 1 - eps, 1 + eps)
    * A) with A the response's advantage; KL term k3 = exp(ref - new) - (ref - new) - 1. The
    per-token loss is -(surrogate - kl_coef * k3). Average over each response's masked tokens,
    then over the G responses. Raise ValueError for the shape, dtype, mask, finiteness and
    parameter rules in problem.md. Never mutate the inputs.
    """
    raise NotImplementedError
