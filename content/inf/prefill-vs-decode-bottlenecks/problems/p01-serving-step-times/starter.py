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
    raise NotImplementedError
