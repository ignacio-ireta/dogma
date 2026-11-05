"""Pack token-id documents into next-token (inputs, targets) blocks."""

from __future__ import annotations

import torch


def pack_next_token_blocks(
    documents: list[list[int]] | tuple[list[int], ...],
    block_size: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets), both int64 of shape (B, block_size).

    The stream is each document followed by eos_id, concatenated. Block b takes inputs
    S[b*T : b*T + T] and targets S[b*T + 1 : b*T + T + 1], with B = (N - 1) // T where N is
    the stream length. Remaining tokens are dropped, never padded. Inputs are not mutated.
    Raises ValueError for invalid arguments.
    """
    raise NotImplementedError
