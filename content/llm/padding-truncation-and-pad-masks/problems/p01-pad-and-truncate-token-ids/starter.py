"""Pad and truncate token-id lists into a batch with an attention mask."""

from __future__ import annotations

import torch


def pad_and_truncate(
    sequences: list[list[int]] | tuple[list[int], ...],
    pad_id: int,
    max_length: int,
    *,
    truncation_side: str = "right",
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, attention_mask), both int64 of shape (B, max_length).

    Each row keeps at most max_length tokens: the first ones when truncation_side is "right",
    the last ones when it is "left". Kept tokens go after the pads when padding_side is "right"
    and before them when it is "left". attention_mask is 1 on kept tokens, 0 on pads, and depends
    on lengths only. Inputs are not mutated. Raises ValueError for invalid arguments.
    """
    raise NotImplementedError
