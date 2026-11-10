from __future__ import annotations

import math
from typing import Any

CONFIG_KEYS = ("num_layers", "num_kv_heads", "head_dim")


def _check_int(name: str, value: Any, minimum: int) -> int:
    # type(...) is int rejects bool (a subclass of int), float, str and None.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def compute_bound_decode_batch(
    params: int,
    config: dict[str, Any],
    context_len: int,
    element_bytes: int,
    *,
    peak_flops: int,
    bandwidth: int,
) -> int:
    """Wrong on purpose: the threshold is a float quotient rounded up with math.ceil. Return the smallest batch whose decode step is compute-bound, or 0 if none is.

    See starter.py for the model and contract.
    """  # noqa: E501
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")
    for key in CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"config is missing the required key {key}")
    params = _check_int("params", params, 1)
    layers = _check_int("num_layers", config["num_layers"], 1)
    kv_heads = _check_int("num_kv_heads", config["num_kv_heads"], 1)
    head_dim = _check_int("head_dim", config["head_dim"], 1)
    context = _check_int("context_len", context_len, 0)
    width = _check_int("element_bytes", element_bytes, 1)
    peak = _check_int("peak_flops", peak_flops, 1)
    bw = _check_int("bandwidth", bandwidth, 1)

    # Step 1: the bytes that do not depend on the batch (the weights), and the bytes each
    # sequence adds to the step: its cached tokens plus the new one, at kv bytes per token.
    weight_bytes = params * width
    kv_per_token = 2 * layers * kv_heads * head_dim * width
    per_sequence = (context + 1) * kv_per_token

    # Step 2: compute-bound means 2 * params * B * bw >= peak * (weight_bytes + B * per_sequence).
    # Both sides are linear in B, so the condition is B * slope >= needed with
    #   slope  = 2 * params * bw - peak * per_sequence
    #   needed = peak * weight_bytes.
    # Only a positive slope can ever satisfy it, since needed is positive.
    slope = 2 * params * bw - peak * per_sequence
    if slope <= 0:
        return 0

    # Step 3: the smallest integer B is the ceiling of needed / slope, computed exactly with
    # integers. A float division would round a threshold like 2**70 + 0.5 down. At least 1.
    return max(1, math.ceil(peak * weight_bytes / slope))
