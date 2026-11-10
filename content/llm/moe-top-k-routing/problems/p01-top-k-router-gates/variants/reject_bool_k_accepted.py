from __future__ import annotations

import torch


def top_k_route(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrong on purpose: bool k is accepted as an int. Route each token to its top-k experts and return the gates and per-expert counts.

    Shapes: logits (T, E) float64, k a Python int with 1 <= k <= E. Returns
    (expert_indices (T, k) int64, gate_weights (T, k) float64, expert_counts (E,) int64).
    """  # noqa: E501
    # Validate the interface first. A bool is an int in Python, so it is rejected by name.
    if not isinstance(k, int):
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

    # Step 1, selection. A stable descending sort keeps equal logits in their original order,
    # so ties land on the lower expert index. The sort uses the logits, not the probabilities,
    # because softmax is monotone in the logits and the logits stay exact when probabilities
    # underflow to zero.
    order = torch.sort(logits, dim=-1, descending=True, stable=True).indices  # (T, E)
    expert_indices = order[:, :k].contiguous()  # (T, k): slot j is the j-th ranked expert

    # Step 2, gates. Gather the chosen logits and take a softmax over the k of them. This equals
    # the full-row softmax probability of each chosen expert divided by the sum of the chosen
    # probabilities (the renormalisation step). The softmax subtracts the max, so large logits
    # cannot overflow.
    chosen_logits = torch.gather(logits, 1, expert_indices)  # (T, k)
    gate_weights = torch.softmax(chosen_logits, dim=-1)  # rows sum to 1

    # Step 3, load. Every (token, slot) pair counts, not only the first slot. bincount over the
    # flattened indices gives one count per expert; minlength keeps experts with no tokens.
    expert_counts = torch.bincount(expert_indices.reshape(-1), minlength=E).to(torch.int64)

    return expert_indices, gate_weights, expert_counts
