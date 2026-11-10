from __future__ import annotations

from collections.abc import Callable

import torch


def _is_int(value: object) -> bool:
    # bool is a subclass of int in Python, but a flag is not a token id or a step count.
    return isinstance(value, int) and not isinstance(value, bool)


def greedy_generate(
    next_logits: Callable[[torch.Tensor], torch.Tensor],
    prompts: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    """Wrong on purpose: Greedy decoding for a batch of equal-length prompts, with per-row EOS and padding.

    Each step is: score the prefix, take the argmax per row, then mark rows that hit EOS.
    """  # noqa: E501
    if not isinstance(prompts, torch.Tensor) or prompts.ndim != 2 or prompts.dtype != torch.int64:
        raise ValueError("prompts must be a 2-D int64 tensor of shape (B, P)")
    B, P = prompts.shape
    if B < 1 or P < 1:
        raise ValueError("prompts must have at least one row and one token")
    if not _is_int(max_new_tokens) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative int")
    if not _is_int(eos_id) or not _is_int(pad_id):
        raise ValueError("eos_id and pad_id must be ints")

    # Work on a copy: the caller's prompts are never written to, and the output never aliases them.
    tokens = prompts.clone()
    # One flag per row. Once True it stays True: the row only emits pad_id from now on.
    finished = prompts[:, -1] == eos_id

    for _ in range(max_new_tokens):
        # Stop as soon as every row has emitted EOS. This is the batch-level stopping rule.
        if bool(finished.all()):
            break

        # The model sees the whole prefix, including pad_id columns for finished rows.
        logits = next_logits(tokens)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != B:
            raise ValueError("next_logits must return a 2-D tensor of shape (B, V)")
        if logits.shape[1] < 1:
            raise ValueError("next_logits must return at least one vocabulary entry")

        # Greedy choice: argmax over the vocabulary for every row. argmax returns the first
        # maximum, so ties go to the lowest token id.
        choice = torch.argmax(logits, dim=-1)

        # Finished rows ignore their logits and append pad_id. The tensor stays rectangular.
        next_tok = torch.where(finished, torch.full_like(choice, pad_id), choice)
        tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)

        # A row that just emitted EOS is finished from the next step on. The EOS token itself
        # stays in the output, so the caller can see where the answer ended.
        finished = finished | ((next_tok == eos_id) & ~finished)

    return tokens
