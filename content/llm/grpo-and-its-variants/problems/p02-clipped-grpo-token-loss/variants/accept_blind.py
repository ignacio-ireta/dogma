from __future__ import annotations

import math

import torch


def _is_number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def grpo_token_loss(  # noqa: PLR0912
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> torch.Tensor:
    tensors = {
        "logp_new": logp_new,
        "logp_old": logp_old,
        "logp_ref": logp_ref,
        "advantages": advantages,
        "mask": mask,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")

    if logp_new.dim() != 2:
        raise ValueError("logp_new must be 2-D with shape (G, T)")
    G, T = logp_new.shape
    if G == 0 or T == 0:
        raise ValueError("G and T must be non-zero")
    for name in ("logp_old", "logp_ref"):
        if tensors[name].shape != (G, T):
            raise ValueError(f"{name} must have shape (G, T)")
    if advantages.dim() != 1 or advantages.shape[0] != G:
        raise ValueError("advantages must have shape (G,)")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if mask.dim() != 2 or mask.shape != (G, T):
        raise ValueError("mask must have shape (G, T)")

    dtype = logp_new.dtype
    if not torch.is_floating_point(logp_new):
        raise ValueError("log-probabilities must be floating-point")
    for name in ("logp_old", "logp_ref", "advantages"):
        if tensors[name].dtype != dtype:
            raise ValueError(f"{name} dtype differs from logp_new")
    for name in ("logp_new", "logp_old", "logp_ref", "advantages"):
        if not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"{name} must be finite")

    if not (_is_number(clip_eps) and 0 < clip_eps < 1):
        raise ValueError("clip_eps must lie in (0, 1)")
    if not (_is_number(kl_coef) and math.isfinite(kl_coef) and kl_coef >= 0):
        raise ValueError("kl_coef must be finite and >= 0")

    if not bool(mask.any(dim=1).all()):
        raise ValueError("every response must have at least one response token")

    zeros = torch.zeros((G, T), dtype=dtype)
    # Sanitise masked positions first so extreme values there cannot produce inf or NaN.
    diff_new_old = torch.where(mask, logp_new - logp_old, zeros)
    diff_ref_new = torch.where(mask, logp_ref - logp_new, zeros)

    ratio = torch.exp(diff_new_old)
    adv = advantages.unsqueeze(1)  # (G, 1), broadcast over tokens
    surrogate = torch.minimum(
        ratio * adv,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    )
    # k3 = exp(ref - new) - (ref - new) - 1
    k3 = torch.exp(diff_ref_new) - diff_ref_new - 1.0

    per_token = -(surrogate - kl_coef * k3)
    per_token = torch.where(mask, per_token, zeros)

    per_response_sum = per_token.sum(dim=1)  # (G,)
    counts = mask.sum(dim=1).to(dtype)  # (G,)
    per_response_mean = per_response_sum / counts
    return per_response_mean.mean()
