from __future__ import annotations

import math

import torch


def _check_scalar(
    name: str, value: float, *, low_open: bool, high_open: bool, upper: float | None
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if low_open and not value > 0:
        raise ValueError(f"{name} must be greater than 0")
    if not low_open and value < 0:
        raise ValueError(f"{name} must be >= 0")
    if upper is not None and not value < upper:
        raise ValueError(f"{name} must be less than {upper}")


def sgd_nesterov_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (new_params, new_velocity) after one Nesterov SGD step with coupled weight decay."""
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    if velocity is not None and len(velocity) != len(params):
        raise ValueError("velocity must have one buffer per parameter")
    _check_scalar("lr", lr, low_open=False, high_open=False, upper=None)
    _check_scalar("weight_decay", weight_decay, low_open=False, high_open=False, upper=None)
    _check_scalar("momentum", momentum, low_open=True, high_open=True, upper=1.0)

    new_params: list[torch.Tensor] = []
    new_velocity: list[torch.Tensor] = []
    for i, (p, g) in enumerate(zip(params, grads)):  # noqa: B905
        if g.shape != p.shape or g.dtype != p.dtype:
            raise ValueError("each grad must match its parameter's shape and dtype")
        if not p.is_floating_point():
            raise ValueError("parameters must be floating-point tensors")
        if velocity is not None and velocity[i].shape != p.shape:
            raise ValueError("each velocity buffer must match its parameter's shape")

        d = g + weight_decay * p
        if velocity is None:
            v = d.clone()
        else:
            v = momentum * velocity[i] + d
        d_nes = d + momentum * v
        new_params.append(p - lr * d_nes)
        new_velocity.append(v)
    return new_params, new_velocity
