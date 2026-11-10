from __future__ import annotations

from typing import Any

# Bytes per element for each dtype name. Only these four are accepted.
BYTES_PER_ELEMENT = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
REQUIRED_KEYS = ("num_layers", "num_kv_heads", "head_dim")


def _check_int(name: str, value: Any, minimum: int) -> int:
    # type(...) is int rejects bool (a subclass of int) as well as float, str and None.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def kv_cache_bytes(
    config: dict[str, Any],
    seq_len: int,
    batch: int = 1,
    dtype: str = "fp16",
) -> int:
    """Wrong on purpose: the byte count is returned as a float.

    See starter.py for the full contract.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"config is missing the required key {key}")

    layers = _check_int("num_layers", config["num_layers"], 1)
    kv_heads = _check_int("num_kv_heads", config["num_kv_heads"], 1)
    head_dim = _check_int("head_dim", config["head_dim"], 1)
    window = config.get("sliding_window")  # None or missing means full attention
    if window is not None:
        window = _check_int("sliding_window", window, 1)
    seq = _check_int("seq_len", seq_len, 0)
    count = _check_int("batch", batch, 1)

    if type(dtype) is not str:
        raise TypeError(f"dtype must be a str, got {type(dtype).__name__}")
    if dtype not in BYTES_PER_ELEMENT:
        raise ValueError(f"dtype must be one of {sorted(BYTES_PER_ELEMENT)}, got {dtype!r}")
    element_bytes = BYTES_PER_ELEMENT[dtype]

    # Step 1: tokens actually stored per layer. A sliding window caps the context; it does not
    # scale the cache. With no window every token of the context is kept.
    stored_tokens = seq if window is None else min(seq, window)

    # Step 2: the factor 2 is K and V. Each KV head stores one key and one value of width head_dim
    # for every stored token, in every layer, for every sequence in the batch.
    return 2 * layers * kv_heads * head_dim * stored_tokens * count * element_bytes / 1
