from __future__ import annotations

import math

import torch


def _check_grid(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Shape, dtype, mask and finiteness rules for the (G, T) grid and the (G,) advantages."""
    if logp_new.ndim != 2:
        raise ValueError("log-probabilities must have shape (G, T)")
    groups, length = logp_new.shape
    if groups == 0 or length == 0:
        raise ValueError("at least one response with one token is required")
    if logp_old.shape != logp_new.shape or logp_ref.shape != logp_new.shape:
        raise ValueError("logp_old and logp_ref must have the same shape as logp_new")
    if advantages.shape != (groups,):
        raise ValueError("advantages must have shape (G,): one value per response")
    if mask.shape != logp_new.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor with the shape (G, T)")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every response needs at least one masked-in token")
    dtypes = {logp_new.dtype, logp_old.dtype, logp_ref.dtype, advantages.dtype}
    if len(dtypes) != 1 or not logp_new.is_floating_point():
        raise ValueError("log-probabilities and advantages must share one floating-point dtype")
    finite = (
        torch.isfinite(logp_new).all()
        & torch.isfinite(logp_old).all()
        & torch.isfinite(logp_ref).all()
        & torch.isfinite(advantages).all()
    )
    if not bool(finite):
        raise ValueError("log-probabilities and advantages must be finite")


def _check_scalars(clip_eps: object, kl_coef: object) -> None:
    """Range rules for the clip width and the KL coefficient."""
    if isinstance(clip_eps, bool) or not isinstance(clip_eps, float | int):
        raise ValueError("clip_eps must be a real number")
    if not 0.0 < float(clip_eps) < 1.0:
        raise ValueError("clip_eps must lie strictly between 0 and 1")
    if isinstance(kl_coef, bool) or not isinstance(kl_coef, float | int):
        raise ValueError("kl_coef must be a real number")
    if not math.isfinite(float(kl_coef)) or float(kl_coef) < 0.0:
        raise ValueError("kl_coef must be finite and non-negative")


def grpo_token_loss(  # noqa: PLR0917
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> torch.Tensor:
    """Wrong on purpose: clip_eps can be passed by position, though the Contract says keyword-only"""  # noqa: E501
    if not all(
        isinstance(t, torch.Tensor) for t in (logp_new, logp_old, logp_ref, advantages, mask)
    ):
        raise ValueError("all five arguments must be torch tensors")
    _check_grid(logp_new, logp_old, logp_ref, advantages, mask)
    _check_scalars(clip_eps, kl_coef)
    # Step 1: the probability ratio of each sampled token, new policy over the sampling policy.
    ratio = torch.exp(logp_new - logp_old)
    # Step 2: each response's advantage applies to all of its tokens. Shape (G, 1).
    adv = advantages.unsqueeze(1)
    # Step 3: clipped surrogate. The min keeps the pessimistic branch, so a token whose ratio
    # has moved beyond the clip range in the advantage's direction stops receiving gradient.
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    # Step 4: the k3 KL estimator to the reference. Non-negative for every sample.
    delta = logp_ref - logp_new
    k3 = torch.exp(delta) - delta - 1.0
    # Step 5: the per-token loss is the negative of the penalised objective.
    per_token = -(surrogate - kl_coef * k3)
    # Step 6: where, not multiplication, removes masked positions. A masked token can have a huge
    # log-ratio whose exponential is inf, and inf * 0 would be NaN.
    kept = torch.where(mask, per_token, torch.zeros_like(per_token))
    # Step 7: average each response over its own response tokens, then average the group.
    tokens_per_response = mask.sum(dim=1).to(per_token.dtype)
    per_response = kept.sum(dim=1) / tokens_per_response
    return per_response.mean()
