"""Wrong on purpose: empty result float dtype."""

from __future__ import annotations

import torch

INT64_MAX = 2**63 - 1


def _check_int(name: str, value: object, *, minimum: int) -> int:
    """Block size and EOS id are plain ints; bool subclasses int, so it is rejected explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a Python int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _check_document(doc: object) -> list[int]:
    """Each document is a list or tuple of token ids that fit in int64."""
    if not isinstance(doc, (list, tuple)):
        raise ValueError("each document must be a list or tuple of token ids")
    for token in doc:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("token ids must be Python ints")
        if token < 0 or token > INT64_MAX:
            raise ValueError("token ids must lie in [0, 2**63 - 1]")
    return list(doc)  # a copy, so the caller's lists are never changed


def pack_next_token_blocks(
    documents: list[list[int]] | tuple[list[int], ...],
    block_size: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets), both int64 of shape (B, block_size)."""
    if not isinstance(documents, (list, tuple)):
        raise ValueError("documents must be a list or tuple of token-id lists")
    T = _check_int("block_size", block_size, minimum=1)
    eos = _check_int("eos_id", eos_id, minimum=0)
    docs = [_check_document(doc) for doc in documents]

    # Lesson "The system", rule 1: each document is followed by one EOS separator.
    stream: list[int] = []
    for doc in docs:
        stream.extend(doc)
        stream.append(eos)
    n = len(stream)

    # Lesson rule 3: a block needs T + 1 tokens, the window. Full windows start every T tokens,
    # so the count is (N - 1) // T. Leftover tokens are dropped, never padded.
    n_blocks = max(0, (n - 1) // T)
    if n_blocks == 0:
        empty = torch.zeros((0, T))
        return empty, empty.clone()

    # Lesson rule 2: block b's window is S[b*T : b*T + T + 1]. Inputs are its first T tokens,
    # targets its last T, so the targets are the inputs shifted left by one.
    window = torch.tensor(stream, dtype=torch.int64)
    starts = torch.arange(n_blocks, dtype=torch.int64) * T  # (B,)
    offsets = torch.arange(T + 1, dtype=torch.int64)  # (T+1,)
    index = starts[:, None] + offsets[None, :]  # (B, T+1), window positions
    windows = window[index]
    inputs = windows[:, :T].contiguous()
    targets = windows[:, 1:].contiguous()
    return inputs, targets
