from __future__ import annotations

import math

import torch


def _validate(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    lr: float,
    momentum: float,
) -> None:
    """Reject bad inputs before any arithmetic, so errors name the broken contract rule."""
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
    if not (math.isfinite(momentum) and 0 <= momentum < 1):
        raise ValueError("momentum must lie in [0, 1)")


def sgd_momentum_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    lr: float,
    momentum: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (new_params, new_velocity) after one SGD step with heavy-ball momentum.

    Per tensor: v = g on the first step, v = momentum * v_prev + g afterwards, then
    p_new = p - lr * v. The buffer v carries no learning rate; lr enters only the update.
    """
    _validate(params, grads, velocity, lr, momentum)
    new_params: list[torch.Tensor] = []
    new_velocity: list[torch.Tensor] = []
    for index, (p, g) in enumerate(zip(params, grads, strict=True)):
        if velocity is None:
            # First step: the buffer is seeded with the raw gradient. Seeding it with
            # (1 - momentum) * g would shrink the very first move by that factor.
            buf = g.clone()
        else:
            # Exponential moving sum of past gradients: old direction decays by momentum.
            buf = g + momentum * velocity[index]
        new_velocity.append(buf)
        # Only the parameter update sees the learning rate. Out-of-place ops keep inputs intact.
        new_params.append(p - lr * buf)
    return new_params, new_velocity
