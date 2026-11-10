from __future__ import annotations

_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}
_OPTIMIZER_STATES = {"none": 0, "sgd": 0, "momentum": 1, "adam": 2, "adamw": 2}
_MODES = ("inference", "training")


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
    if isinstance(num_params, bool) or not isinstance(num_params, int) or num_params < 0:
        raise ValueError("num_params must be an int at least 0")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError('mode must be "inference" or "training"')
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise ValueError("unknown dtype")
    if not isinstance(optimizer, str) or optimizer not in _OPTIMIZER_STATES:
        raise ValueError("unknown optimizer")
    if not isinstance(master_weights, bool):
        raise ValueError("master_weights must be a bool")
    if isinstance(activation_bytes, bool) or not isinstance(activation_bytes, int):
        raise ValueError("activation_bytes must be an int")
    if activation_bytes < 0:
        raise ValueError("activation_bytes must be at least 0")

    if master_weights and dtype == "fp32":
        raise ValueError("master_weights needs a lower-precision dtype; fp32 has no master copy")

    training = mode == "training"
    if not training:
        if optimizer != "none":
            raise ValueError('inference holds no optimizer state: use optimizer="none"')
        if master_weights:
            raise ValueError("inference holds no master copy: master_weights must be False")
    elif optimizer == "none":
        raise ValueError('training needs an optimizer other than "none"')

    bytes_per = _DTYPE_BYTES[dtype]
    weights = num_params * bytes_per
    gradients = num_params * bytes_per if training else 0
    master = 4 * num_params if (training and master_weights and dtype != "fp32") else 0
    states = 4 * _OPTIMIZER_STATES[optimizer] * num_params if training else 0
    activations = activation_bytes
    total = weights + gradients + master + states + activations
    return {
        "weights": weights,
        "gradients": gradients,
        "master_weights": master,
        "optimizer_states": states,
        "activations": activations,
        "total": total,
    }
