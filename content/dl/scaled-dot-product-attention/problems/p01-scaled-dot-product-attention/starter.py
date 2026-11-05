from __future__ import annotations

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output, weights) for softmax(q @ k^T / sqrt(d_k)) @ v with an optional mask.

    q: floating (*batch, L, d_k); k: (*batch, S, d_k); v: (*batch, S, d_v), same dtype.
    mask: None or bool broadcastable to (*batch, L, S); True means the pair may attend.
    Weights are (*batch, L, S) with rows summing to 1 over allowed keys. A query row with no
    allowed key gives zero weights and zero output, never NaN. Raise ValueError on invalid input.
    Never mutate inputs; the result must be finite for large-magnitude scores.
    """
    raise NotImplementedError
