from __future__ import annotations

from typing import Any


def compute_bound_decode_batch(
    params: int,
    config: dict[str, Any],
    context_len: int,
    element_bytes: int,
    *,
    peak_flops: int,
    bandwidth: int,
) -> int:
    """Return the smallest batch whose decode step is compute-bound, or 0 if none is.

    One decode step for B sequences with context_len cached tokens reads W = params * element_bytes
    weight bytes and (context_len + 1) * kv bytes per sequence, where
    kv = 2 * num_layers * num_kv_heads * head_dim * element_bytes. It performs 2 * params * B FLOPs.
    The step is compute-bound when FLOPs * bandwidth >= peak_flops * bytes (a tie is compute).
    Solve that inequality for B with integers. Return 0 when no batch size satisfies it.

    Raises TypeError for a value that is not a Python int (or config not a dict) and ValueError
    for a value below its minimum (context_len >= 0, others >= 1) or a missing config key.
    """
    raise NotImplementedError
