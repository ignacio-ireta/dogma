from __future__ import annotations

from typing import Any


def min_gpu_count(  # noqa: PLR0917 (the contract fixes six positional arguments)
    params: int,
    config: dict[str, Any],
    batch: int,
    context_len: int,
    element_bytes: int,
    gpu_memory_bytes: int,
    *,
    workspace_bytes: int,
    reserve_percent: int,
) -> int:
    """Return the smallest power-of-two tensor-parallel degree whose per-GPU total fits, or 0.

    Allowed degrees are 1, 2, 4, ... that divide num_kv_heads. usable = gpu_memory_bytes *
    (100 - reserve_percent) // 100. Per-GPU weights are ceil(params * element_bytes / T). Per-GPU
    KV bytes are 2 * num_layers * (num_kv_heads // T) * head_dim * stored * batch * element_bytes,
    with stored = min(context_len, sliding_window) when a window is set, else context_len. The total
    is weights + KV + workspace_bytes, and a total equal to usable fits. Validate every argument
    (TypeError for non-int, bool, float, str, None, or a non-dict config; ValueError for a value out
    of range, a missing key, or reserve_percent outside 0 to 99).
    """
    raise NotImplementedError
