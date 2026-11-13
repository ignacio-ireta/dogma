from __future__ import annotations

from typing import Any

CONFIG_KEYS = ("num_layers", "num_kv_heads", "head_dim")


def _check_int(name: str, value: Any, minimum: int) -> int:
    # type(...) is int rejects bool (a subclass of int), float, str and None.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


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

    See starter.py for the model and contract.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")
    for key in CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"config is missing the required key {key}")
    params = _check_int("params", params, 1)
    layers = _check_int("num_layers", config["num_layers"], 1)
    kv_heads = _check_int("num_kv_heads", config["num_kv_heads"], 1)
    head_dim = _check_int("head_dim", config["head_dim"], 1)
    window = config.get("sliding_window")  # None or missing means full attention
    if window is not None:
        window = _check_int("sliding_window", window, 1)
    count = _check_int("batch", batch, 1)
    context = _check_int("context_len", context_len, 1)
    width = _check_int("element_bytes", element_bytes, 1)
    memory = _check_int("gpu_memory_bytes", gpu_memory_bytes, 1)
    workspace = _check_int("workspace_bytes", workspace_bytes, 0)
    reserve = _check_int("reserve_percent", reserve_percent, 0)
    if reserve > 99:
        raise ValueError(f"reserve_percent must be at most 99, got {reserve}")

    # Step 1: the budget a GPU can really use. The reserve is a whole percent, floored.
    usable = memory * (100 - reserve) // 100

    # Step 2: the cache tokens each sequence stores. A window caps the context, not scales it.
    stored = context if window is None else min(context, window)
    weight_bytes = params * width

    # Step 3: try the degrees 1, 2, 4, ... in order. A degree is allowed only if it divides
    # num_kv_heads, so each KV head sits on one GPU and the KV split is exact.
    degree = 1
    while degree <= kv_heads:
        if kv_heads % degree == 0:
            # Step 4: each GPU holds a ceiling share of the weights, and 1/T of the KV heads.
            weights_per_gpu = -(-weight_bytes // degree)
            kv_per_gpu = 2 * layers * (kv_heads // degree) * head_dim * stored * count
            kv_per_gpu *= width
            total = weights_per_gpu + kv_per_gpu + workspace
            if total <= usable:
                return degree
        degree *= 2
    return 0


_correct = min_gpu_count


def _change_in_place(v):
    v["probe"] = 0


def min_gpu_count(*args, **kwargs):
    """Wrong on purpose: the result is right, but this overwrites `config` in place."""
    result = _correct(*args, **kwargs)
    target = args[1] if len(args) > 1 else kwargs.get("config")
    if type(target).__name__ in ("dict",):
        _change_in_place(target)
    return result


min_gpu_count.__wrapped__ = _correct
