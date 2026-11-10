"""Pad and truncate token-id lists into a batch with an attention mask."""

from __future__ import annotations

import torch

_MAX_ID = 2**63 - 1


def _check_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}")
    return value


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
    if not isinstance(sequences, (list, tuple)):
        raise ValueError("sequences must be a list or tuple")
    _check_int(pad_id, "pad_id", 0)
    _check_int(max_length, "max_length", 1)
    if truncation_side not in ("right", "left") or padding_side not in ("right", "left"):
        raise ValueError("sides must be 'right' or 'left'")

    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for row in sequences:
        if not isinstance(row, (list, tuple)):
            raise ValueError("each row must be a list or tuple")
        for token in row:
            if type(token) is not int or token < 0 or token > _MAX_ID:
                raise ValueError("token ids must be Python ints in 0..2**63-1")

        kept = list(row)
        if len(kept) > max_length:
            kept = kept[:max_length] if truncation_side == "right" else kept[-max_length:]
        k = len(kept)
        pad_count = max_length - k
        if padding_side == "right":
            ids = kept + [pad_id] * pad_count
            mask = [1] * k + [0] * pad_count
        else:
            ids = [pad_id] * pad_count + kept
            mask = [0] * pad_count + [1] * k
        input_rows.append(ids)
        mask_rows.append(mask)

    input_ids = torch.tensor(input_rows, dtype=torch.int64).reshape(len(input_rows), max_length)
    attention_mask = torch.tensor(mask_rows, dtype=torch.int64).reshape(len(mask_rows), max_length)
    return input_ids, attention_mask
