from __future__ import annotations

import torch


def causal_padding_mask(
    query_len: int,
    key_len: int,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a boolean mask, True = may attend, bottom-right aligned to the keys.

    Query i may attend key j when j <= key_len - query_len + i. If key_valid (bool, shape
    (N, key_len), True = real key) is given, the result has shape (N, L, S) and is ANDed with it;
    otherwise the result has shape (L, S). Fully masked rows are all False, never NaN, never an
    error. Raise ValueError for invalid sizes or a non-boolean or misshapen key_valid.
    Never mutate key_valid.
    """
    raise NotImplementedError
