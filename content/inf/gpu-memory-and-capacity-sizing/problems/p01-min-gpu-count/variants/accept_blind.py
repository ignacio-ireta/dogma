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
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    def _check_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a Python int")

    for name, value in (
        ("params", params),
        ("batch", batch),
        ("context_len", context_len),
        ("element_bytes", element_bytes),
        ("gpu_memory_bytes", gpu_memory_bytes),
        ("workspace_bytes", workspace_bytes),
        ("reserve_percent", reserve_percent),
    ):
        _check_int(name, value)
    for key in ("num_layers", "num_kv_heads", "head_dim"):
        if key not in config:
            raise ValueError(f"config is missing the key {key!r}")
        _check_int(key, config[key])
    window = config.get("sliding_window")
    if window is not None:
        _check_int("sliding_window", window)

    for name, value, minimum in (
        ("params", params, 1),
        ("batch", batch, 1),
        ("context_len", context_len, 1),
        ("element_bytes", element_bytes, 1),
        ("gpu_memory_bytes", gpu_memory_bytes, 1),
        ("workspace_bytes", workspace_bytes, 0),
        ("reserve_percent", reserve_percent, 0),
        ("num_layers", config["num_layers"], 1),
        ("num_kv_heads", config["num_kv_heads"], 1),
        ("head_dim", config["head_dim"], 1),
    ):
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if reserve_percent > 99:
        raise ValueError("reserve_percent must be at most 99")
    if window is not None and window < 1:
        raise ValueError("sliding_window must be at least 1")

    usable = gpu_memory_bytes * (100 - reserve_percent) // 100
    stored = context_len if window is None else min(context_len, window)
    weight_bytes = params * element_bytes
    num_kv_heads = config["num_kv_heads"]

    degree = 1
    while degree <= num_kv_heads:
        if num_kv_heads % degree == 0:
            weights = -(-weight_bytes // degree)
            kv = (
                2
                * config["num_layers"]
                * (num_kv_heads // degree)
                * config["head_dim"]
                * stored
                * batch
                * element_bytes
            )
            if weights + kv + workspace_bytes <= usable:
                return degree
        degree *= 2
    return 0
