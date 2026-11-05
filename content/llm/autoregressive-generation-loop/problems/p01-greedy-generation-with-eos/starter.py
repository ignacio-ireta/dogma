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
    """Greedy decoding for a batch of equal-length prompts, with per-row EOS and padding.

    next_logits(tokens) takes an int64 (B, T) tensor and returns (B, V) logits for the token that
    follows each row. prompts is an int64 (B, P) tensor with B >= 1 and P >= 1. Each step appends
    the argmax of every unfinished row (ties go to the lowest id). A row is finished after it
    appends eos_id, and it appends pad_id from then on. The loop stops when all rows are finished
    or after max_new_tokens steps. Returns an int64 (B, P + G) tensor, where G is the number of
    steps run; shorter rows are right-padded with pad_id.

    Raises ValueError for a non-int64 or non-2-D prompts tensor, B == 0, P == 0, a negative or
    non-int max_new_tokens, a non-int eos_id or pad_id, or logits that are not (B, V) with V >= 1.
    Never mutates prompts.
    """
    raise NotImplementedError
