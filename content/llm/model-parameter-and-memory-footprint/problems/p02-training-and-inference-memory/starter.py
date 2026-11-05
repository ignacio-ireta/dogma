from __future__ import annotations


def estimate_memory(
    num_params: int,
    *,
    mode: str,
    dtype: str,
    optimizer: str,
    master_weights: bool,
    activation_bytes: int,
) -> dict[str, int]:
    """Return the memory terms in bytes: weights, gradients, master_weights, optimizer_states,
    activations and total, as exact Python ints.

    mode: "inference" | "training". dtype: "fp32" (4 bytes), "fp16" or "bf16" (2), "fp8" or "int8"
    (1). optimizer: "none", "sgd" (0 fp32 states), "momentum" (1), "adam" or "adamw" (2).
    Optimizer states are always fp32. master_weights adds 4 bytes per parameter in training with a
    non-fp32 dtype. Raise ValueError for invalid arguments. Use no numeric library.
    """
    raise NotImplementedError
