from __future__ import annotations

from typing import Any

_BYTES_PER_ELEMENT = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def _int(name: str, value: object, minimum: int) -> int:
    if type(value) is not int and not (isinstance(value, int) and not isinstance(value, bool)):
        raise TypeError(f"{name} must be a Python int")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
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
    """Return the largest number of sequences whose KV cache fits after the weights and reserve.

    Parameters
    ----------
    config : dict
        Keys num_layers, num_kv_heads and head_dim (required, Python int >= 1), and
        sliding_window (None, or a Python int >= 1). Other keys are ignored. Not mutated.
    seq_len : int
        Tokens of context per sequence, a Python int >= 1.
    dtype : str
        One of "fp32" (4 bytes), "fp16" (2), "bf16" (2) or "fp8" (1).
    weight_bytes : int
        Bytes taken by the model weights, a Python int >= 0.
    memory_bytes : int
        Device memory in bytes, a Python int >= 0.
    reserve_bytes : int
        Bytes kept back for activations and overhead, a Python int >= 0.

    Returns
    -------
    int
        The largest B >= 0 with
        weight_bytes + reserve_bytes + B * per_sequence_bytes <= memory_bytes,
        where per_sequence_bytes is the KV-cache size of one sequence. 0 when nothing fits.

    Raises
    ------
    TypeError
        When a value that must be an int is not a Python int, or config is not a dict.
    ValueError
        When a required key is missing, a value is out of range, or dtype is unknown.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    for key in ("num_layers", "num_kv_heads", "head_dim"):
        if key not in config:
            raise ValueError(f"config is missing required key {key!r}")

    num_layers = _int("num_layers", config["num_layers"], 1)
    num_kv_heads = _int("num_kv_heads", config["num_kv_heads"], 1)
    head_dim = _int("head_dim", config["head_dim"], 1)
    window = config.get("sliding_window")
    if window is not None:
        window = _int("sliding_window", window, 1)

    seq_len = _int("seq_len", seq_len, 1)
    weight_bytes = _int("weight_bytes", weight_bytes, 0)
    memory_bytes = _int("memory_bytes", memory_bytes, 0)
    reserve_bytes = _int("reserve_bytes", reserve_bytes, 0)

    if not isinstance(dtype, str):
        raise TypeError("dtype must be a str")
    if dtype not in _BYTES_PER_ELEMENT:
        raise ValueError(f"unknown dtype {dtype!r}")
    bytes_per_element = _BYTES_PER_ELEMENT[dtype]

    stored_tokens = seq_len if window is None else min(seq_len, window)
    per_sequence = 2 * num_layers * num_kv_heads * head_dim * stored_tokens * bytes_per_element

    budget = memory_bytes - weight_bytes - reserve_bytes
    if budget < per_sequence:
        return 0
    return budget // per_sequence
