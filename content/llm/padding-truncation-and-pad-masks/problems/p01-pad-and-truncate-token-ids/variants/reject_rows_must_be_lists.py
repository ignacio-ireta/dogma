"""Wrong on purpose: rows must be lists, so valid tuple rows are refused."""

from __future__ import annotations

import torch

INT64_MAX = 2**63 - 1


def _check_int(name: str, value: object, *, minimum: int) -> int:
    """Sizes and pad ids are plain ints. bool subclasses int, so it is rejected explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a Python int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _check_side(name: str, value: object) -> str:
    """Only the two literal side names are accepted; anything else is a caller error."""
    if not isinstance(value, str) or value not in ("left", "right"):
        raise ValueError(f"{name} must be the string 'left' or 'right'")
    return value


def _check_row(row: object) -> list[int]:
    """Each row must be a list or tuple of token ids that fit in int64."""
    if not isinstance(row, list):
        raise ValueError("each sequence must be a list or tuple of token ids")
    for token in row:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("token ids must be Python ints")
        if token < 0 or token > INT64_MAX:
            raise ValueError("token ids must lie in [0, 2**63 - 1]")
    return list(row)


def pad_and_truncate(
    sequences: list[list[int]] | tuple[list[int], ...],
    pad_id: int,
    max_length: int,
    *,
    truncation_side: str = "right",
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, attention_mask), both int64 of shape (B, max_length)."""
    if not isinstance(sequences, (list, tuple)):
        raise ValueError("sequences must be a list or tuple of token-id lists")
    pad = _check_int("pad_id", pad_id, minimum=0)
    width = _check_int("max_length", max_length, minimum=1)
    trunc_side = _check_side("truncation_side", truncation_side)
    pad_side = _check_side("padding_side", padding_side)

    # Validate every row before allocating output, so a bad row never yields a partial batch.
    rows = [_check_row(row) for row in sequences]

    input_ids = torch.full((len(rows), width), pad, dtype=torch.int64)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.int64)

    for r, tokens in enumerate(rows):
        # Lesson "The system", step 1: truncate on the side named by truncation_side.
        # A slice of a copied list never changes the caller's input.
        if len(tokens) > width:
            kept = tokens[:width] if trunc_side == "right" else tokens[len(tokens) - width :]
        else:
            kept = tokens
        k = len(kept)  # number of real positions in this row

        # Lesson step 2: pads fill the rest of the row. The kept tokens start at column 0 for
        # right padding, and at column width - k for left padding.
        start = 0 if pad_side == "right" else width - k
        if k > 0:
            input_ids[r, start : start + k] = torch.tensor(kept, dtype=torch.int64)

        # Lesson step 3: the mask follows the length. Columns that hold kept tokens get 1,
        # whatever the token id is. A real token equal to pad_id is still a real token.
        attention_mask[r, start : start + k] = 1

    return input_ids, attention_mask
