from __future__ import annotations


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
    raise NotImplementedError
