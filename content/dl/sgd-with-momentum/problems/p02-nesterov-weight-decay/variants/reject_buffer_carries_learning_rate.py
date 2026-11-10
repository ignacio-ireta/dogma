from __future__ import annotations

import math

import torch


def _validate(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> None:
    """Reject bad inputs before arithmetic so each error names the broken rule."""
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    if velocity is not None and len(velocity) != len(params):
        raise ValueError("velocity must hold one buffer per parameter")
    for index, (p, g) in enumerate(zip(params, grads, strict=True)):
        if not isinstance(p, torch.Tensor) or not isinstance(g, torch.Tensor):
            raise ValueError("params and grads must be torch tensors")
        if p.shape != g.shape:
            raise ValueError("each gradient must have the shape of its parameter")
        if velocity is not None and velocity[index].shape != p.shape:
            raise ValueError("each velocity buffer must have the shape of its parameter")
    if not (math.isfinite(lr) and lr >= 0):
        raise ValueError("lr must be a finite non-negative number")
    if not (math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError("weight_decay must be a finite non-negative number")
    # Nesterov look-ahead is defined only with a nonzero momentum that is still below one.
    if not (math.isfinite(momentum) and 0 < momentum < 1):
        raise ValueError("momentum must lie in (0, 1) for Nesterov")


def sgd_nesterov_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Wrong on purpose: the returned buffer already includes the learning rate

    Return (new_params, new_velocity) after one Nesterov step with coupled L2 decay."""
    _validate(params, grads, velocity, lr=lr, momentum=momentum, weight_decay=weight_decay)
    new_params: list[torch.Tensor] = []
    new_velocity: list[torch.Tensor] = []
    for index, (p, g) in enumerate(zip(params, grads, strict=True)):
        # Coupled weight decay: the penalty gradient is added before momentum sees it.
        d = g + weight_decay * p
        if velocity is None:
            buf = d.clone()  # first step seeds the buffer with the decayed gradient
        else:
            buf = momentum * velocity[index] + d
        # Nesterov: take the gradient at the look-ahead point, which is one momentum move
        # further along the buffer than the current iterate.
        look_ahead = d + momentum * buf
        new_velocity.append(lr * buf)
        new_params.append(p - lr * look_ahead)
    return new_params, new_velocity
