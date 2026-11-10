from __future__ import annotations

from typing import Any

_BYTES_PER_ELEMENT = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def _required_int(config: dict[str, Any], key: str, minimum: int) -> int:
    if key not in config:
        raise ValueError(f"config is missing required key {key!r}")
    value = config[key]
    if type(value) is not int:
        raise TypeError(f"config[{key!r}] must be a Python int")
    if value < minimum:
        raise ValueError(f"config[{key!r}] must be at least {minimum}")
    return value


def kv_cache_bytes(
    config: dict[str, Any],
    seq_len: int,
    batch: int = 1,
    dtype: str = "fp16",
) -> int:
    """Return the exact number of bytes a KV cache occupies.

    Parameters
    ----------
    config : dict
        Keys num_layers, num_kv_heads and head_dim (required, Python int >= 1), and
        sliding_window (None, or a Python int >= 1). Other keys are ignored. Not mutated.
    seq_len : int
        Tokens of context per sequence, a Python int >= 0.
    batch : int
        Number of sequences, a Python int >= 1.
    dtype : str
        One of "fp32" (4 bytes), "fp16" (2), "bf16" (2) or "fp8" (1).

    Returns
    -------
    int
        2 * num_layers * num_kv_heads * head_dim * min(seq_len, window) * batch * bytes_per_element,
        where window is sliding_window, or seq_len when there is no window.

    Raises
    ------
    TypeError
        When config is not a dict, or seq_len, batch or a config value is not a Python int.
    ValueError
        When a required key is missing, a value is below its minimum, or dtype is unknown.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    if type(seq_len) is not int:
        raise TypeError("seq_len must be a Python int")
    if type(batch) is not int:
        raise TypeError("batch must be a Python int")

    num_layers = _required_int(config, "num_layers", 1)
    num_kv_heads = _required_int(config, "num_kv_heads", 1)
    head_dim = _required_int(config, "head_dim", 1)

    window: int | None = config.get("sliding_window")
    if window is not None:
        if type(window) is not int:
            raise TypeError("sliding_window must be None or a Python int")
        if window < 1:
            raise ValueError("sliding_window must be at least 1")

    if seq_len < 0:
        raise ValueError("seq_len must not be negative")
    if batch < 1:
        raise ValueError("batch must be at least 1")
    if not isinstance(dtype, str) or dtype not in _BYTES_PER_ELEMENT:
        raise ValueError("dtype must be one of fp32, fp16, bf16, fp8")
    bytes_per_element = _BYTES_PER_ELEMENT[dtype]

    stored_tokens = seq_len if window is None else min(seq_len, window)
    return 2 * num_layers * num_kv_heads * head_dim * stored_tokens * batch * bytes_per_element
