"""Masked mean negative log-likelihood and perplexity from logits."""

from __future__ import annotations

import torch


def masked_nll_and_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_nll, perplexity) as 0-dim tensors with the dtype of logits."""
    if not isinstance(logits, torch.Tensor) or logits.dim() != 3:
        raise ValueError("logits must be a 3-D tensor")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating-point")
    if not isinstance(targets, torch.Tensor) or targets.dim() != 2:
        raise ValueError("targets must be a 2-D tensor")
    if targets.dtype != torch.int64:
        raise ValueError("targets must be torch.int64")
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError("targets must have shape (B, T) matching logits")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("ignore_index must be a Python int")

    vocab = logits.shape[-1]
    try:
        mask = targets != ignore_index
    except (OverflowError, RuntimeError):
        mask = torch.ones_like(targets, dtype=torch.bool)
    count = int(mask.sum())
    if count == 0:
        raise ValueError("every target is ignored; the mean NLL is undefined")
    kept = targets[mask]
    if bool(((kept < 0) | (kept >= vocab)).any()):
        raise ValueError("a non-ignored target is outside [0, V-1]")

    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    row_max = logits.max(dim=-1, keepdim=True).values
    lse = row_max.squeeze(-1) + torch.log(torch.exp(logits - row_max).sum(dim=-1))
    target_logit = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    nll = torch.where(mask, lse - target_logit, torch.zeros_like(lse))

    mean_nll = nll.sum() / count
    perplexity = torch.exp(mean_nll)
    return mean_nll, perplexity
