from __future__ import annotations

import torch


def top_k_route(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route each token to its top-k experts and return the gates and per-expert counts."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a Python int")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.dim() != 2:
        raise ValueError("logits must have shape (T, E)")
    if logits.dtype != torch.float64:
        raise ValueError("logits must be float64")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    num_tokens, num_experts = logits.shape
    if num_experts < 1:
        raise ValueError("there must be at least one expert")
    if not 1 <= k <= num_experts:
        raise ValueError("k must satisfy 1 <= k <= E")

    # Stable sort of the negated logits: largest logit first, ties in index order.
    order = torch.sort(-logits, dim=1, stable=True).indices
    expert_indices = order[:, :k].contiguous()

    # Gates: softmax over the chosen logits equals the renormalised full softmax.
    chosen_logits = torch.gather(logits, 1, expert_indices)
    gate_weights = torch.softmax(chosen_logits, dim=1)

    counts = torch.zeros(num_experts, dtype=torch.int64)
    counts.scatter_add_(
        0, expert_indices.reshape(-1), torch.ones(num_tokens * k, dtype=torch.int64)
    )
    return expert_indices, gate_weights, counts
