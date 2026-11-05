from __future__ import annotations

import torch


def multi_head_self_attention(  # noqa: PLR0917 (the contract fixes eleven arguments)
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    b_q: torch.Tensor,
    b_k: torch.Tensor,
    b_v: torch.Tensor,
    b_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the multi-head self-attention output for a batch of sequences.

    Shapes: X (B, L, d_model); each weight (d_model, d_model) in nn.Linear layout, so a
    projection is y = x @ W.T + b; each bias (d_model,). Output (B, L, d_model), float64.

    Heads: d_k = d_model / num_heads. Head j reads the contiguous feature block
    [j*d_k, (j+1)*d_k) of Q, K and V. Scores are scaled by 1/sqrt(d_k).

    Parameters
    ----------
    X : float64 tensor (B, L, d_model), B >= 1 and L >= 1.
    W_q, W_k, W_v, W_o : float64 tensors (d_model, d_model), nn.Linear layout.
    b_q, b_k, b_v, b_o : float64 tensors (d_model,).
    num_heads : positive integer that divides d_model.
    mask : None, or a torch.bool tensor (L, L). mask[i, j] True means query i may attend to key j.
        The same mask applies to all sequences and heads.

    Returns
    -------
    float64 tensor (B, L, d_model). Inputs are not modified.

    Raises
    ------
    ValueError
        On the rank, weight or bias shape rules; a num_heads that does not divide d_model; a mask
        that is not a bool (L, L) tensor; or a mask row with no allowed key.
    """
    raise NotImplementedError
