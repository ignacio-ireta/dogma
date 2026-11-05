"""Masked mean negative log-likelihood and perplexity from logits."""

from __future__ import annotations

import torch


def masked_nll_and_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_nll, perplexity) as 0-dim tensors with the dtype of logits.

    logits has shape (B, T, V); targets is int64 (B, T). Positions whose target equals
    ignore_index are excluded from both the sum and the count. mean_nll is in nats, and
    perplexity = exp(mean_nll). Use a stable log-sum-exp. Inputs are not mutated. Raises
    ValueError for invalid shapes, dtypes, out-of-range targets and an all-ignored batch.
    """
    raise NotImplementedError
