from __future__ import annotations


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ceil_bytes(count: int, bits: int) -> int:
    return -((-count * bits) // 8)


def finetune_memory_bytes(
    *,
    base_params: int,
    base_bits: int,
    lora_params: int,
    adapter_bits: int,
) -> dict[str, int]:
    """Return the bytes of a LoRA fine-tuning state, split into terms, as exact integers.

    Terms: the frozen base at base_bits per parameter; the adapter weights and their gradients
    at adapter_bits per parameter; two fp32 Adam moments (8 bytes) per adapter parameter. Each
    stored tensor is rounded up to whole bytes: ceil(count * bits / 8). Activations are not counted.

    Parameters
    ----------
    base_params : non-negative int, the frozen base parameter count.
    base_bits : one of 4, 8, 16, 32.
    lora_params : non-negative int, the trainable adapter parameter count.
    adapter_bits : one of 16, 32.

    Returns
    -------
    dict with keys base_weights, adapter_weights, adapter_gradients, adam_states, total, in that
    order. total is the exact sum of the other four.

    Raises
    ------
    ValueError
        If an argument is not an int (or is a bool), is negative, or has a
        bit width outside the allowed sets.
    """
    if not _is_int(base_params) or base_params < 0:
        raise ValueError("base_params must be a non-negative int")
    if not _is_int(base_bits) or base_bits not in (4, 8, 16, 32):
        raise ValueError("base_bits must be one of 4, 8, 16, 32")
    if not _is_int(lora_params) or lora_params < 0:
        raise ValueError("lora_params must be a non-negative int")
    if not _is_int(adapter_bits) or adapter_bits not in (16, 32):
        raise ValueError("adapter_bits must be 16 or 32")

    base_weights = _ceil_bytes(base_params, base_bits)
    adapter_weights = _ceil_bytes(lora_params, adapter_bits)
    adapter_gradients = adapter_weights
    adam_states = 8 * lora_params
    total = base_weights + adapter_weights + adapter_gradients + adam_states
    return {
        "base_weights": base_weights,
        "adapter_weights": adapter_weights,
        "adapter_gradients": adapter_gradients,
        "adam_states": adam_states,
        "total": total,
    }
