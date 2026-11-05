from __future__ import annotations

# Bytes per number for each working dtype.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}

# Optimizer states per parameter. Every state is stored in fp32, four bytes each.
OPTIMIZER_STATES = {"none": 0, "sgd": 0, "momentum": 1, "adam": 2, "adamw": 2}

FP32_BYTES = 4


def _int(name: str, value: object) -> int:
    """A count or byte size: a non-negative int, with bool rejected."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def estimate_memory(
    num_params: int,
    *,
    mode: str,
    dtype: str,
    optimizer: str,
    master_weights: bool,
    activation_bytes: int,
) -> dict[str, int]:
    """Return memory terms in bytes for inference or training, as exact Python ints."""
    count = _int("num_params", num_params)
    activations = _int("activation_bytes", activation_bytes)
    if mode not in ("inference", "training"):
        raise ValueError('mode must be "inference" or "training"')
    if dtype not in DTYPE_BYTES:
        raise ValueError("dtype must be one of fp32, fp16, bf16, fp8, int8")
    if optimizer not in OPTIMIZER_STATES:
        raise ValueError("optimizer must be none, sgd, momentum, adam or adamw")
    if type(master_weights) is not bool:
        raise ValueError("master_weights must be a bool")

    bytes_per_value = DTYPE_BYTES[dtype]
    if mode == "inference":
        # Inference holds the weights and nothing else.
        if optimizer != "none":
            raise ValueError('inference holds no optimizer state: use optimizer="none"')
        if master_weights:
            raise ValueError("master_weights applies to training only")
        terms = {
            "weights": count * bytes_per_value,
            "gradients": 0,
            "master_weights": 0,
            "optimizer_states": 0,
        }
    else:
        if optimizer == "none":
            raise ValueError("training needs an optimizer: choose sgd, momentum, adam or adamw")
        if master_weights and dtype == "fp32":
            raise ValueError("master_weights needs a lower-precision dtype: fp32 has no copy")
        terms = {
            # Weights and gradients both use the working dtype.
            "weights": count * bytes_per_value,
            "gradients": count * bytes_per_value,
            # A fp32 copy of each weight, kept by the optimizer, when the working dtype is lower.
            "master_weights": count * FP32_BYTES if master_weights else 0,
            # Each optimizer state is one fp32 number per parameter.
            "optimizer_states": count * FP32_BYTES * OPTIMIZER_STATES[optimizer],
        }

    terms["activations"] = activations
    terms["total"] = sum(terms.values())
    return terms
