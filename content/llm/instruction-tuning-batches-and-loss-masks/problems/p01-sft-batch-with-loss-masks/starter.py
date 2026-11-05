from __future__ import annotations

from collections.abc import Sequence

import torch


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
    raise NotImplementedError
