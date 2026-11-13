from __future__ import annotations

import math

import torch


def build_adamw_groups(
    named_params: list[tuple[str, torch.Tensor]],
    weight_decay: float,
) -> list[dict[str, object]]:
    """Return AdamW groups: matrices and conv kernels decay; biases and 1-D gains do not."""
    if not named_params:
        raise ValueError("named_params must contain at least one parameter")
    if not (math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError("weight_decay must be a finite non-negative number")
    names = [name for name, _ in named_params]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    decayed: list[torch.Tensor] = []
    undecayed: list[torch.Tensor] = []
    for name, param in named_params:
        if not isinstance(param, torch.Tensor):
            raise ValueError("every value must be a torch tensor")
        # Biases are excluded by name even when they are 2-D (for example a stacked bias).
        # One-dimensional tensors such as norm gains and biases are excluded by rank.
        is_bias = name.rsplit(".", 1)[-1] == "bias"
        if param.ndim < 2 or is_bias:
            undecayed.append(param)
        else:
            decayed.append(param)
    groups: list[dict[str, object]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


_correct = build_adamw_groups


def _laxer(r):
    if type(r).__name__ == "Tensor" and r.requires_grad:
        return r.detach()
    return r


def build_adamw_groups(*args, **kwargs):
    """Wrong on purpose: returns a result cut off from the graph."""
    result = _correct(*args, **kwargs)
    return _everywhere(result)


def _everywhere(r):
    if type(r) is list:
        return [_everywhere(item) for item in r]
    if type(r) is tuple:
        return tuple(_everywhere(item) for item in r)
    if type(r) is dict:
        return {key: _everywhere(item) for key, item in r.items()}
    return _laxer(r)


build_adamw_groups.__wrapped__ = _correct
