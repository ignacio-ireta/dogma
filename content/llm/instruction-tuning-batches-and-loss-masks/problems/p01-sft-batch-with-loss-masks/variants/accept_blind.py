from __future__ import annotations

from collections.abc import Sequence

import torch

_IGNORE = -100
_MAX_ID = 2**63 - 1


def _check_id(value: object, what: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{what} must be a Python int")
    if value < 0 or value > _MAX_ID:
        raise ValueError(f"{what} must be in 0..2**63-1")
    return value


def _check_seq(value: object, what: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{what} must be a list or tuple")
    return [_check_id(v, f"{what} token id") for v in value]


def build_sft_batch(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (input_ids, labels, attention_mask) for prompt/response pairs.

    Each pair is (prompt_ids, response_ids), both non-empty lists of Python int ids. Rows are
    prompt + response, right padded with pad_id to the longest row, so every tensor is (B, L).

    labels is aligned to input_ids, not shifted: response positions hold their token id, and every
    prompt position and every padding position holds -100 (the cross-entropy ignore index). The
    one-token shift happens in the loss.

    Masks come from lengths, never from token values: a response token equal to pad_id keeps its
    label and its attention mask of 1.

    Returns
    -------
    input_ids : int64 tensor (B, L)
    labels : int64 tensor (B, L)
    attention_mask : int64 tensor (B, L), 1 on real tokens, 0 on padding

    Raises
    ------
    ValueError
        If pairs is not a list or tuple, an item is not a two-element list or tuple, a prompt or
        response is empty or not a list or tuple, an id is not a non-negative Python int within
        int64, or pad_id is not a valid Python int.
    """
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("pairs must be a list or tuple")

    checked: list[tuple[list[int], list[int]]] = []
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each pair must be a two-element list or tuple")
        prompt = _check_seq(item[0], "prompt")
        response = _check_seq(item[1], "response")
        if len(prompt) == 0:
            raise ValueError("prompt must contain at least one token")
        if len(response) == 0:
            raise ValueError("response must contain at least one token")
        checked.append((prompt, response))

    pad = _check_id(pad_id, "pad_id")

    batch = len(checked)
    if batch == 0:
        empty = torch.zeros((0, 0), dtype=torch.int64)
        return empty, empty.clone(), empty.clone()

    length = max(len(p) + len(r) for p, r in checked)
    input_ids = torch.full((batch, length), pad, dtype=torch.int64)
    labels = torch.full((batch, length), _IGNORE, dtype=torch.int64)
    attention_mask = torch.zeros((batch, length), dtype=torch.int64)

    for i, (prompt, response) in enumerate(checked):
        n_p = len(prompt)
        n = n_p + len(response)
        input_ids[i, :n] = torch.tensor(prompt + response, dtype=torch.int64)
        labels[i, n_p:n] = torch.tensor(response, dtype=torch.int64)
        attention_mask[i, :n] = 1

    return input_ids, labels, attention_mask
