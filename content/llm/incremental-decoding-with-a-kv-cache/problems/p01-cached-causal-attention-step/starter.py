from __future__ import annotations

import torch


def cached_attention_step(  # noqa: PLR0917 (the contract fixes seven arguments)
    x_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One cached causal multi-head attention step.

    x_new is float64 (B, T, d) with B >= 1 and T >= 1. cache is None or (K_c, V_c), each float64
    (B, h, S, d_k) with S >= 0 and d_k = d // h. Weights are float64 (d, d) in nn.Linear layout
    (y = x @ W.T), with no biases. New query i is at absolute position S + i and attends key j
    exactly when j <= S + i.

    Returns (output, new_cache): output is float64 (B, T, d), and new_cache is (K, V), each
    float64 (B, h, S + T, d_k), whose first S positions equal the input cache.

    Raises ValueError for a bad x_new, zero batch or chunk length, num_heads not dividing d, a bad
    weight shape, or a cache that is not None or a (K, V) pair of matching (B, h, S, d_k) tensors.
    Never mutates x_new, the weights or the input cache.
    """
    raise NotImplementedError
