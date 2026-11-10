"""Pack token-id documents into next-token (inputs, targets) blocks."""

from __future__ import annotations

import torch

_MAX_ID = 2**63 - 1


def _check_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}")
    return value


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
    if not isinstance(documents, (list, tuple)):
        raise ValueError("documents must be a list or tuple")
    _check_int(block_size, "block_size", 1)
    _check_int(eos_id, "eos_id", 0)

    stream: list[int] = []
    for document in documents:
        if not isinstance(document, (list, tuple)):
            raise ValueError("each document must be a list or tuple")
        for token in document:
            if type(token) is not int or token < 0 or token > _MAX_ID:
                raise ValueError("token ids must be Python ints in 0..2**63-1")
        stream.extend(document)
        stream.append(eos_id)

    total = len(stream)
    blocks = max(0, (total - 1) // block_size)
    inputs = [stream[b * block_size : b * block_size + block_size] for b in range(blocks)]
    targets = [stream[b * block_size + 1 : b * block_size + block_size + 1] for b in range(blocks)]
    input_ids = torch.tensor(inputs, dtype=torch.int64).reshape(blocks, block_size)
    target_ids = torch.tensor(targets, dtype=torch.int64).reshape(blocks, block_size)
    return input_ids, target_ids
