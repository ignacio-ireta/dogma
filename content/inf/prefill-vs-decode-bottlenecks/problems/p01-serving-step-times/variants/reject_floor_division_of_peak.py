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
    """Wrong on purpose: the decode compute floor uses floor division. Estimate prefill time, one decode step and decode throughput.

    See starter.py for the full model and contract.
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
    prompt = _check_int("prompt_len", prompt_len, 1)
    context = _check_int("context_len", context_len, 0)
    count = _check_int("batch", batch, 1)
    width = _check_int("element_bytes", element_bytes, 1)
    peak = _check_int("peak_flops", peak_flops, 1)
    bw = _check_int("bandwidth", bandwidth, 1)

    # Step 1: the two fixed byte quantities. The weights are read once per pass, whatever the batch.
    weight_bytes = params * width
    # Step 2: KV bytes per token per sequence. The leading 2 is K and V, for every layer and head.
    kv_per_token = 2 * layers * kv_heads * head_dim * width

    # Step 3: prefill processes batch * prompt_len tokens in one parallel pass, so the weights are
    # read once and the work is 2 FLOPs per parameter per token.
    prefill_flops = 2 * params * prompt * count
    prefill_bytes = weight_bytes + count * prompt * kv_per_token
    prefill_s = max(prefill_flops / peak, prefill_bytes / bw)

    # Step 4: a decode step gives one token per sequence. The work is 2 FLOPs per parameter
    # per sequence, but the weights are still read once, and each sequence reads its cache
    # (the cached tokens plus the new one).
    step_flops = 2 * params * count
    step_bytes = weight_bytes + count * (context + 1) * kv_per_token
    decode_step_s = max(step_flops // peak, step_bytes / bw)

    # Step 5: one step yields one token for each of the batch sequences.
    return {
        "prefill_s": prefill_s,
        "decode_step_s": decode_step_s,
        "decode_tokens_per_s": count / decode_step_s,
    }
