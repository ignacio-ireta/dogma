from __future__ import annotations

import torch


def build_adamw_groups(
    named_params: list[tuple[str, torch.Tensor]],
    weight_decay: float,
) -> list[dict[str, object]]:
    """Split named parameters into AdamW parameter groups: decayed weights and undecayed rest.

    A parameter is decayed when it has at least two dimensions and its final dotted name
    component is not ``"bias"``. Everything else (biases, 1-D norm gains) is not decayed.
    Return the decayed group first, with ``{"params": [...], "weight_decay": weight_decay}``,
    then the undecayed group with ``weight_decay`` 0.0. Keep parameter order within each group.
    Omit a group that would be empty. The dicts must hold the caller's tensor objects, not copies.
    Raise ``ValueError`` for an empty list, duplicate names, non-tensor values, or a
    non-finite or negative ``weight_decay``.
    """
    raise NotImplementedError
