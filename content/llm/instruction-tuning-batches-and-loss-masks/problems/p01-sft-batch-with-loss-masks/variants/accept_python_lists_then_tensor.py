from __future__ import annotations

from collections.abc import Sequence

import torch

IGNORE_INDEX = -100  # PyTorch cross_entropy's default ignore_index
_INT64_MAX = 2**63 - 1


def _check_ids(seq: object, name: str) -> list[int]:
    # A token id must be a plain non-negative int that fits in int64. bool is a subclass of int,
    # so it is rejected explicitly; a bool id would silently become 0 or 1.
    if not isinstance(seq, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of token ids")
    if len(seq) == 0:
        raise ValueError(f"{name} must contain at least one token")
    ids: list[int] = []
    for tok in seq:
        if isinstance(tok, bool) or not isinstance(tok, int):
            raise ValueError(f"{name} ids must be Python int, not {type(tok).__name__}")
        if tok < 0 or tok > _INT64_MAX:
            raise ValueError(f"{name} ids must lie in 0..2**63-1")
        ids.append(tok)
    return ids


def build_sft_batch(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (input_ids, labels, attention_mask) for prompt/response pairs (see problem.md)."""
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("pairs must be a list or tuple of (prompt, response) pairs")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or not 0 <= pad_id <= _INT64_MAX:
        raise ValueError("pad_id must be a Python int in 0..2**63-1")

    # Validate every example first, so a bad row fails before any tensor is built.
    rows: list[tuple[list[int], list[int]]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each item of pairs must be a (prompt, response) pair")
        prompt = _check_ids(pair[0], "prompt")
        response = _check_ids(pair[1], "response")
        rows.append((prompt, response))

    # Build each row as a plain Python list, then convert once. The same batch, another route.
    B = len(rows)
    if B == 0:
        empty = torch.zeros((0, 0), dtype=torch.int64)
        return empty, empty.clone(), empty.clone()
    L = max(len(p) + len(r) for p, r in rows)

    ids_rows, lab_rows, mask_rows = [], [], []
    for prompt, response in rows:
        n = len(prompt) + len(response)
        pad_tail = L - n
        ids_rows.append(prompt + response + [pad_id] * pad_tail)
        lab_rows.append([IGNORE_INDEX] * len(prompt) + response + [IGNORE_INDEX] * pad_tail)
        mask_rows.append([1] * n + [0] * pad_tail)

    input_ids = torch.tensor(ids_rows, dtype=torch.int64)
    labels = torch.tensor(lab_rows, dtype=torch.int64)
    attention_mask = torch.tensor(mask_rows, dtype=torch.int64)
    return input_ids, labels, attention_mask
