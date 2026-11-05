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

    # Step 1, the rectangle. Every row has n_i = len(prompt) + len(response) real positions;
    # the batch is as wide as the longest row, so B x L = (number of rows) x max(n_i).
    B = len(rows)
    L = max((len(p) + len(r) for p, r in rows), default=0)

    # Start from the padding values: pad_id for inputs, -100 for labels, 0 for the mask.
    # Filling the three tensors this way means padding columns need no extra step.
    input_ids = torch.full((B, L), pad_id, dtype=torch.int64)
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.int64)
    attention_mask = torch.zeros((B, L), dtype=torch.int64)

    for i, (prompt, response) in enumerate(rows):
        n = len(prompt) + len(response)
        seq = prompt + response

        # Step 2, inputs. The prompt then the response fill columns 0..n-1, left to right.
        input_ids[i, :n] = torch.tensor(seq, dtype=torch.int64)

        # Step 3, the loss mask. Only the response columns keep their ids. The prompt columns
        # stay -100, so the model is never trained to reproduce the user's text. The labels are
        # NOT shifted: column t holds the id of input token t, and the loss moves it one step.
        p = len(prompt)
        labels[i, p:n] = torch.tensor(response, dtype=torch.int64)

        # Step 4, the attention mask comes from the length n, not from the token values.
        attention_mask[i, :n] = 1

    return input_ids, labels, attention_mask
