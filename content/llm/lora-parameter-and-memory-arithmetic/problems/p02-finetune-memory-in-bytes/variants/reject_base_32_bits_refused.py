from __future__ import annotations

_BASE_BITS = (4, 8, 16)
_ADAPTER_BITS = (16, 32)
_ADAM_BYTES_PER_PARAM = 2 * 4  # two fp32 moments, four bytes each, whatever the adapter dtype


def _count(value: object, name: str) -> int:
    # bool is a subclass of int, so a bool count would silently act as 0 or 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _bytes_for(count: int, bits: int) -> int:
    # Round up to whole bytes with integer arithmetic only: ceil(count * bits / 8).
    # Adding 7 before the floor division lifts any fractional byte to the next whole byte.
    return (count * bits + 7) // 8


def finetune_memory_bytes(
    *,
    base_params: int,
    base_bits: int,
    lora_params: int,
    adapter_bits: int,
) -> dict[str, int]:
    """Wrong on purpose: refuses a 32-bit base."""
    base_params = _count(base_params, "base_params")
    lora_params = _count(lora_params, "lora_params")
    # Membership alone is not enough: 16.0 == 16 is True in Python, so the type is checked too.
    if isinstance(base_bits, bool) or not isinstance(base_bits, int) or base_bits not in _BASE_BITS:
        raise ValueError("base_bits must be one of 4, 8, 16, 32")
    if (
        isinstance(adapter_bits, bool)
        or not isinstance(adapter_bits, int)
        or adapter_bits not in _ADAPTER_BITS
    ):
        raise ValueError("adapter_bits must be 16 or 32")

    # Step 1, the frozen base. It needs storage only: no gradient and no optimizer state.
    base_weights = _bytes_for(base_params, base_bits)

    # Step 2, the trainable adapter. Weights and gradients share the adapter dtype.
    adapter_weights = _bytes_for(lora_params, adapter_bits)
    adapter_gradients = adapter_weights

    # Step 3, Adam keeps two fp32 tensors per trainable number, whatever the adapter dtype is.
    adam_states = lora_params * _ADAM_BYTES_PER_PARAM

    # Step 4, the exact sum. Activations are not part of the state and are not added.
    total = base_weights + adapter_weights + adapter_gradients + adam_states
    return {
        "base_weights": base_weights,
        "adapter_weights": adapter_weights,
        "adapter_gradients": adapter_gradients,
        "adam_states": adam_states,
        "total": total,
    }
