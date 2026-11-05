from __future__ import annotations

from collections.abc import Callable

import torch


def train_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lr: float,
) -> float:
    """Run one manual SGD step on the batch (x, y) and return the pre-update loss.

    Order: clear each parameter's .grad, forward, scalar loss, backward, then update every
    parameter in place under torch.no_grad() as p <- p - lr * p.grad. Parameters without a gradient
    stay unchanged. Returns float(loss) computed before the update.
    Raises ValueError if lr is not a finite number > 0 (bool is invalid) or loss_fn returns a
    non-scalar tensor. Must not mutate x or y. No torch.optim.
    """
    raise NotImplementedError
