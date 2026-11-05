from __future__ import annotations

import torch


def sgd_nesterov_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    velocity: list[torch.Tensor] | None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (new_params, new_velocity) after one Nesterov SGD step with coupled weight decay.

    Per tensor: ``d = g + weight_decay * p`` (decay joins the gradient before momentum);
    ``v = d`` on the first step (``velocity is None``), otherwise ``v = momentum * v_prev + d``;
    look-ahead ``d_nes = d + momentum * v``; then ``p_new = p - lr * d_nes``.
    The returned velocity is the unscaled buffer ``v``.
    ``params`` and ``grads`` are equal-length lists of matching-shape float tensors.
    ``lr`` and ``weight_decay`` are finite and >= 0; ``momentum`` is in (0, 1).
    Never mutate the inputs. Raise ``ValueError`` on bad arguments or shapes.
    """
    raise NotImplementedError
