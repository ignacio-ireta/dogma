from __future__ import annotations

import math

import torch


def build_adamw_groups(
    named_params: list[tuple[str, torch.Tensor]],
    weight_decay: float,
) -> list[dict[str, object]]:
    """Split named parameters into AdamW parameter groups: decayed weights and undecayed rest."""
    if not isinstance(named_params, list) or len(named_params) == 0:
        raise ValueError("named_params must be a non-empty list")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise ValueError("weight_decay must be a number")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and >= 0")

    seen: set[str] = set()
    decayed: list[torch.Tensor] = []
    undecayed: list[torch.Tensor] = []
    for item in named_params:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("each entry must be a (name, tensor) pair")
        name, tensor = item
        if not isinstance(name, str):
            raise ValueError("parameter names must be strings")
        if name in seen:
            raise ValueError(f"duplicate parameter name: {name}")
        seen.add(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"value for {name} is not a tensor")
        last = name.split(".")[-1]
        if tensor.dim() >= 2 and last != "bias":
            decayed.append(tensor)
        else:
            undecayed.append(tensor)

    groups: list[dict[str, object]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups
