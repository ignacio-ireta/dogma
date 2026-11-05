from __future__ import annotations

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
    raise NotImplementedError
