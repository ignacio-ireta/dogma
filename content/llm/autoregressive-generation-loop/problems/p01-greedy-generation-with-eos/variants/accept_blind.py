from __future__ import annotations

from collections.abc import Callable

import torch


def greedy_generate(
    next_logits: Callable[[torch.Tensor], torch.Tensor],
    prompts: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    if not isinstance(prompts, torch.Tensor) or prompts.dtype != torch.int64 or prompts.dim() != 2:
        raise ValueError("prompts must be a 2-D int64 tensor")
    batch, plen = prompts.shape
    if batch == 0 or plen == 0:
        raise ValueError("prompts must have at least one row and one column")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an int")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    for value in (eos_id, pad_id):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("eos_id and pad_id must be ints")

    tokens = prompts.clone()
    finished = torch.zeros(batch, dtype=torch.bool)
    steps = 0
    while steps < max_new_tokens and not bool(finished.all()):
        logits = next_logits(tokens)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.dim() != 2
            or logits.shape[0] != batch
            or logits.shape[1] < 1
        ):
            raise ValueError("next_logits must return a 2-D tensor of shape (B, V) with V >= 1")
        picked = torch.argmax(logits, dim=1).to(torch.int64)
        nxt = torch.where(finished, torch.full_like(picked, pad_id), picked)
        finished = finished | (nxt == eos_id)
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        steps += 1
    return tokens
