from __future__ import annotations

import torch


def top_k_route(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route each token to its top-k experts and return the gates and per-expert counts.

    Shapes: logits (T, E) float64, finite, T >= 0, E >= 1. k is a Python int, 1 <= k <= E.
    Ties in logit go to the lower expert index. The gates of each token are the softmax
    probabilities of its k chosen experts, renormalised to sum to 1. Must not mutate logits.

    Returns
    -------
    expert_indices : (T, k) int64, slot j of token t is its j-th ranked expert.
    gate_weights   : (T, k) float64, each row sums to 1.
    expert_counts  : (E,) int64, number of (token, slot) assignments per expert.

    Raises TypeError for a non-int or bool k, and ValueError for a wrong rank or dtype,
    a non-finite logit, or k outside 1..E.
    """
    raise NotImplementedError
