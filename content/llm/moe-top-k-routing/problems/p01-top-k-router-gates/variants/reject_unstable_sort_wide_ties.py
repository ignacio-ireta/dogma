from __future__ import annotations

import torch


def top_k_route(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route each token to its top-k experts and return the gates and per-expert counts.

    Shapes: logits (T, E) float64, k a Python int with 1 <= k <= E. Returns
    (expert_indices (T, k) int64, gate_weights (T, k) float64, expert_counts (E,) int64).
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a Python int")
    if logits.ndim != 2:
        raise ValueError("logits must be 2-D (T, E)")
    if logits.dtype != torch.float64:
        raise ValueError("logits must be float64")
    E = logits.shape[1]
    if not 1 <= k <= E:
        raise ValueError("k must satisfy 1 <= k <= E")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    order = torch.sort(logits, dim=-1, descending=True, stable=False).indices
    expert_indices = order[:, :k].contiguous()
    chosen_logits = torch.gather(logits, 1, expert_indices)
    gate_weights = torch.softmax(chosen_logits, dim=-1)
    expert_counts = torch.bincount(expert_indices.reshape(-1), minlength=E).to(torch.int64)
    return (expert_indices, gate_weights, expert_counts)
