from __future__ import annotations

from typing import Any


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
    raise NotImplementedError
