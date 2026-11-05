from __future__ import annotations

import torch


def grouped_query_attention(  # noqa: PLR0917 (the contract fixes seven arguments)
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the grouped-query self-attention output for a batch of sequences.

    Shapes: X (B, L, d_model); W_q and W_o (d_model, d_model); W_k and W_v (n_kv_heads * d_head,
    d_model); all in nn.Linear layout, so a projection is y = x @ W.T. There are no biases.
    d_head = d_model / n_heads. Query head h reads key and value head h // group_size, where
    group_size = n_heads / n_kv_heads.

    Parameters
    ----------
    X : float64 tensor (B, L, d_model), B >= 1 and L >= 1.
    W_q, W_k, W_v, W_o : float64 tensors in nn.Linear layout (see the shapes above).
    n_heads : positive int. n_kv_heads : positive int that divides n_heads.
    mask : None, or a torch.bool tensor (L, L). mask[i, j] True means query i may attend to key j.

    Returns
    -------
    float64 tensor (B, L, d_model). Inputs are not modified.

    Raises
    ------
    ValueError
        On a wrong rank, weight shape, head count (not positive, not dividing, or not dividing
        d_model), or a mask that is not a bool (L, L) tensor or has a row with no allowed key.
    """
    raise NotImplementedError
