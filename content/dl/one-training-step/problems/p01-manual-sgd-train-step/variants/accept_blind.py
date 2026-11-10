from __future__ import annotations

import math
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
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite number greater than zero")

    params = list(model.parameters())
    for p in params:
        p.grad = None

    prediction = model(x)
    loss = loss_fn(prediction, y)
    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        raise ValueError("loss_fn must return a 0-dimensional tensor")

    loss.backward()
    loss_value = float(loss)

    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.sub_(lr * p.grad)

    return loss_value
