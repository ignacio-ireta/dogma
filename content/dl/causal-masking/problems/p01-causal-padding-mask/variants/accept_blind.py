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
    for name, value in (("query_len", query_len), ("key_len", key_len)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    if key_valid is not None:
        if not isinstance(key_valid, torch.Tensor) or key_valid.dtype != torch.bool:
            raise ValueError("key_valid must be a boolean tensor")
        if key_valid.dim() != 2 or key_valid.shape[1] != key_len:
            raise ValueError("key_valid must have shape (N, key_len)")

    rows = torch.arange(query_len, dtype=torch.long).unsqueeze(1)
    cols = torch.arange(key_len, dtype=torch.long).unsqueeze(0)
    causal = cols <= (key_len - query_len) + rows

    if key_valid is None:
        return causal.clone()
    return causal.unsqueeze(0) & key_valid.unsqueeze(1)
