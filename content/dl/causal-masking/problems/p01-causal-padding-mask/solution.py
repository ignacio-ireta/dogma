from __future__ import annotations

import torch


def _check_size(name: str, value: object) -> int:
    """Sizes are plain non-negative integers; bool is excluded even though it subclasses int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a Python int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def causal_padding_mask(
    query_len: int,
    key_len: int,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the bottom-right aligned causal mask, optionally ANDed with a key padding mask."""
    L = _check_size("query_len", query_len)
    S = _check_size("key_len", key_len)
    # Lesson "The system", rule 1: query i sits at absolute position S - L + i, the last query at
    # the last key. It may attend key j when j <= S - L + i. Note the offset can be negative
    # (L > S), which leaves the early query rows with no allowed key.
    i = torch.arange(L).unsqueeze(1)  # (L, 1)
    j = torch.arange(S).unsqueeze(0)  # (1, S)
    causal = j <= (S - L) + i  # (L, S), boolean
    if key_valid is None:
        return causal.clone()
    # Lesson rule 2: padding is a property of keys (columns), so it broadcasts over the query
    # axis. A key is usable only if both the causal rule and the padding indicator allow it.
    if not isinstance(key_valid, torch.Tensor) or key_valid.dtype != torch.bool:
        raise ValueError("key_valid must be a boolean tensor (True = real key)")
    if key_valid.ndim != 2 or key_valid.shape[1] != S:
        raise ValueError("key_valid must have shape (N, key_len)")
    # The result is a new tensor, so key_valid is never written to.
    return causal.unsqueeze(0) & key_valid.unsqueeze(1)  # (N, L, S)
