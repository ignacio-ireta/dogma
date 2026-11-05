from __future__ import annotations

from typing import Any


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
    raise NotImplementedError
