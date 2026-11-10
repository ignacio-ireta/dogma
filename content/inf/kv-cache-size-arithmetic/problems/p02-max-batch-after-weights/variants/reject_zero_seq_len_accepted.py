from __future__ import annotations

from typing import Any

BYTES_PER_ELEMENT = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
REQUIRED_KEYS = ("num_layers", "num_kv_heads", "head_dim")


def _check_int(name: str, value: Any, minimum: int) -> int:
    # type(...) is int rejects bool, float, str and None.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def max_batch_size(
    config: dict[str, Any],
    seq_len: int,
    dtype: str,
    weight_bytes: int,
    memory_bytes: int,
    *,
    reserve_bytes: int = 0,
) -> int:
    """Wrong on purpose: a seq_len of zero is accepted.

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
    window = config.get("sliding_window")
    if window is not None:
        window = _check_int("sliding_window", window, 1)
    seq = _check_int("seq_len", seq_len, 0)
    weights = _check_int("weight_bytes", weight_bytes, 0)
    memory = _check_int("memory_bytes", memory_bytes, 0)
    reserve = _check_int("reserve_bytes", reserve_bytes, 0)
    if type(dtype) is not str or dtype not in BYTES_PER_ELEMENT:
        raise ValueError(f"dtype must be one of {sorted(BYTES_PER_ELEMENT)}, got {dtype!r}")

    # Step 1: the budget left for caches after the weights and the reserve. Weights larger than
    # memory leave a negative budget, which means no sequence fits at all.
    budget = memory - weights - reserve
    if budget < 0:
        return 0

    # Step 2: the KV-cache bytes of ONE sequence. A window caps the stored tokens per layer, so
    # a context longer than the window costs no more than the window does.
    stored = seq if window is None else min(seq, window)
    per_sequence = 2 * layers * kv_heads * head_dim * stored * BYTES_PER_ELEMENT[dtype]

    # Step 3: a floor division. A partial sequence does not fit, so the answer is the whole
    # number of sequences the budget holds. An exact fit counts.
    return budget // per_sequence
