from __future__ import annotations

from typing import Any


def serving_step_times(  # noqa: PLR0917 (the contract fixes six positional arguments)
    params: int,
    config: dict[str, Any],
    prompt_len: int,
    context_len: int,
    batch: int,
    element_bytes: int,
    *,
    peak_flops: int,
    bandwidth: int,
) -> dict[str, float]:
    """Estimate prefill time, one decode step and decode throughput for a dense decoder.

    Weight bytes W = params * element_bytes, and KV bytes per token per sequence
    kv = 2 * num_layers * num_kv_heads * head_dim * element_bytes. Prefill of batch prompts of
    prompt_len tokens: FLOPs 2 * params * prompt_len * batch, bytes W + batch * prompt_len * kv.
    One decode step for batch sequences with context_len cached tokens: FLOPs 2 * params * batch,
    bytes W + batch * (context_len + 1) * kv. Each phase time is max(FLOPs / peak_flops,
    bytes / bandwidth). Decode throughput is batch / decode_step_s. Ignore sliding_window.

    Raises TypeError for a value that is not a Python int (or config not a dict) and ValueError
    for a value below its minimum (prompt_len >= 1, context_len >= 0, others >= 1) or a missing key.

    Returns a dict with float keys "prefill_s", "decode_step_s" and "decode_tokens_per_s".
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    def _check_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a Python int")

    for name, value in (
        ("params", params),
        ("prompt_len", prompt_len),
        ("context_len", context_len),
        ("batch", batch),
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
        ("prompt_len", prompt_len, 1),
        ("context_len", context_len, 0),
        ("batch", batch, 1),
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

    prefill_flops = 2 * params * prompt_len * batch
    prefill_bytes = weight_bytes + batch * prompt_len * kv
    prefill_s = max(prefill_flops / peak_flops, prefill_bytes / bandwidth)

    decode_flops = 2 * params * batch
    decode_bytes = weight_bytes + batch * (context_len + 1) * kv
    decode_step_s = max(decode_flops / peak_flops, decode_bytes / bandwidth)

    return {
        "prefill_s": float(prefill_s),
        "decode_step_s": float(decode_step_s),
        "decode_tokens_per_s": float(batch / decode_step_s),
    }
