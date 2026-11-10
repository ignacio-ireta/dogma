from __future__ import annotations

import math

import torch


def sgd_momentum_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    lr: float,
    momentum: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (new_params, new_velocity) after one SGD step with heavy-ball momentum.

    Per tensor, with buffer ``v`` and gradient ``g``: ``v = g`` on the first step
    (``velocity is None``), otherwise ``v = momentum * v_prev + g``; then
    ``p_new = p - lr * v``. The returned velocity is the unscaled buffer ``v``.
    ``params`` and ``grads`` are equal-length lists of matching-shape float tensors.
    ``lr`` is finite and >= 0; ``momentum`` is in [0, 1). Never mutate the inputs.
    Raise ``ValueError`` for mismatched lengths, shapes, or invalid scalars.
    """
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and >= 0")
    if not math.isfinite(momentum) or not (0 <= momentum < 1):
        raise ValueError("momentum must be finite and in [0, 1)")
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    for p, g in zip(params, grads):  # noqa: B905
        if p.shape != g.shape:
            raise ValueError("each gradient must have the shape of its parameter")
    if velocity is not None:
        if len(velocity) != len(params):
            raise ValueError("velocity must have one buffer per parameter")
        for p, v in zip(params, velocity):  # noqa: B905
            if p.shape != v.shape:
                raise ValueError("each velocity buffer must have the shape of its parameter")

    new_params: list[torch.Tensor] = []
    new_velocity: list[torch.Tensor] = []
    for i, (p, g) in enumerate(zip(params, grads)):  # noqa: B905
        if velocity is None:
            v = g.clone()
        else:
            v = momentum * velocity[i] + g
        new_velocity.append(v)
        new_params.append(p - lr * v)
    return new_params, new_velocity
