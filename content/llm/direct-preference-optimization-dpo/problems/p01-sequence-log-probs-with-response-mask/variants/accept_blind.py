from __future__ import annotations

import torch

_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.dim() != 3:
        raise ValueError("logits must be a 3-D tensor")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating-point")
    B, T, V = logits.shape
    if B == 0:
        raise ValueError("batch must be non-empty")

    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in _INT_DTYPES:
        raise ValueError("input_ids must be an integer tensor")
    if tuple(input_ids.shape) != (B, T):
        raise ValueError("input_ids must have shape (B, T)")

    if not isinstance(response_mask, torch.Tensor) or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be a boolean tensor")
    if tuple(response_mask.shape) != (B, T):
        raise ValueError("response_mask must have shape (B, T)")
    if bool(response_mask[:, 0].any()):
        raise ValueError("response_mask must be False at position 0 (no prediction for it)")

    if bool(((input_ids < 0) | (input_ids >= V)).any()):
        raise ValueError("input_ids values must lie in [0, V)")

    # Logits at position t - 1 score the token at position t.
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    targets = input_ids[:, 1:].unsqueeze(-1).long()  # (B, T-1, 1)
    picked = logp.gather(-1, targets).squeeze(-1)  # (B, T-1)
    mask = response_mask[:, 1:]
    picked = torch.where(mask, picked, torch.zeros_like(picked))
    return picked.sum(dim=1)
