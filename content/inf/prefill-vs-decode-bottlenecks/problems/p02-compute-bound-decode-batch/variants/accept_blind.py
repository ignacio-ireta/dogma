from __future__ import annotations

from typing import Any


def compute_bound_decode_batch(
    params: int,
    config: dict[str, Any],
    context_len: int,
    element_bytes: int,
    *,
    peak_flops: int,
    bandwidth: int,
) -> int:
    """Return the smallest batch whose decode step is compute-bound, or 0 if none is.

    One decode step for B sequences with context_len cached tokens reads W = params * element_bytes
    weight bytes and (context_len + 1) * kv bytes per sequence, where
    kv = 2 * num_layers * num_kv_heads * head_dim * element_bytes. It performs 2 * params * B FLOPs.
    The step is compute-bound when FLOPs * bandwidth >= peak_flops * bytes (a tie is compute).
    Solve that inequality for B with integers. Return 0 when no batch size satisfies it.

    Raises TypeError for a value that is not a Python int (or config not a dict) and ValueError
    for a value below its minimum (context_len >= 0, others >= 1) or a missing config key.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    def _check_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a Python int")

    for name, value in (
        ("params", params),
        ("context_len", context_len),
        ("element_bytes", element_bytes),
        ("peak_flops", peak_flops),
        ("bandwidth", bandwidth),
    ):
        _check_int(name, value)
    for key in ("num_layers", "num_kv_heads", "head_dim"):
        if key not in config:
            raise ValueError(f"config is missing the key {key!r}")
        _check_int(key, config[key])

    for name, value, minimum in (
        ("params", params, 1),
        ("context_len", context_len, 0),
        ("element_bytes", element_bytes, 1),
        ("peak_flops", peak_flops, 1),
        ("bandwidth", bandwidth, 1),
        ("num_layers", config["num_layers"], 1),
        ("num_kv_heads", config["num_kv_heads"], 1),
        ("head_dim", config["head_dim"], 1),
    ):
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    weight_bytes = params * element_bytes
    kv = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * element_bytes

    # Compute-bound at batch B: 2*params*B*bandwidth >= peak_flops * (W + B*(context_len+1)*kv).
    # Rearranged: B * slope >= peak_flops * W, with slope = 2*params*bandwidth - peak_flops*(context_len+1)*kv.  # noqa: E501
    slope = 2 * params * bandwidth - peak_flops * (context_len + 1) * kv
    if slope <= 0:
        return 0
    numerator = peak_flops * weight_bytes
    return max(1, -(-numerator // slope))
