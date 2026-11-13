"""Reference solution: stable log-sum-exp, a masked sum over real targets, then exp of the mean."""

from __future__ import annotations

import torch


def _check_inputs(logits: torch.Tensor, targets: torch.Tensor, ignore_index: object) -> int:
    """Validate shapes and dtypes before any arithmetic runs. Returns the vocabulary size V."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be a floating-point tensor of shape (B, T, V)")
    if logits.shape[0] == 0 or logits.shape[1] == 0 or logits.shape[2] == 0:
        raise ValueError("logits must have at least one sequence, position and class")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64:
        raise ValueError("targets must be a torch.int64 tensor")
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError("targets must have shape (B, T), matching the first two logits axes")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("ignore_index must be a Python int")
    return int(logits.shape[2])


def masked_nll_and_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_nll, perplexity) as 0-dim tensors with the dtype of logits."""
    vocab = _check_inputs(logits, targets, ignore_index)

    # Lesson "The system": the mask marks positions whose target is scored.
    mask = targets != ignore_index  # (B, T), bool
    if not bool(mask.any()):
        raise ValueError("every target is ignored, so the mean NLL is undefined")
    scored = targets[mask]
    if bool(((scored < 0) | (scored >= vocab)).any()):
        raise ValueError("a non-ignored target lies outside the vocabulary range [0, V-1]")

    # Lesson rule 2: stable log-sum-exp. Subtracting the row maximum keeps every exponent <= 0,
    # so exp cannot overflow; the maximum is added back after the log.
    row_max = logits.max(dim=-1, keepdim=True).values  # (B, T, 1)
    shifted = logits - row_max  # (B, T, V), each entry <= 0
    log_sum_exp = row_max.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))  # (B, T)

    # The target's own logit, one per position. Clamp ignored targets to a valid index only so
    # gather is defined; those positions are removed by the mask below.
    safe_targets = targets.clamp(0, vocab - 1).unsqueeze(-1)  # (B, T, 1)
    target_logit = logits.gather(-1, safe_targets).squeeze(-1)  # (B, T)

    # Lesson rule 1: per-token NLL = LSE(z) - z_y, summed over scored positions only, then
    # divided by the count of scored positions (not B * T, not B).
    nll = log_sum_exp - target_logit  # (B, T)
    count = mask.sum().to(logits.dtype)
    mean_nll = (nll * mask.to(logits.dtype)).sum() / count

    # Lesson rule 3: perplexity is exp of the mean NLL in nats.
    perplexity = torch.exp(mean_nll)
    return mean_nll, perplexity


_correct = masked_nll_and_perplexity


def _laxer(r):
    if type(r).__name__ == "Tensor" and r.requires_grad:
        return r.detach()
    return r


def masked_nll_and_perplexity(*args, **kwargs):
    """Wrong on purpose: returns a result cut off from the graph."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


masked_nll_and_perplexity.__wrapped__ = _correct
