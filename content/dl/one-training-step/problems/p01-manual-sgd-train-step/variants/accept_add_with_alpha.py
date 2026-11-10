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
    """
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite number greater than zero")
    params = list(model.parameters())
    # 1. Clear the previous step's gradients. backward() adds into .grad, so without this line
    #    the update would use the sum of every gradient seen so far (the stale-gradient bug).
    for p in params:
        p.grad = None
    # 2. Forward pass on the current parameters, then the scalar loss for this batch.
    prediction = model(x)
    loss = loss_fn(prediction, y)
    if loss.ndim != 0:
        raise ValueError("loss_fn must return a 0-dimensional (scalar) tensor")
    # 3. Backward: each parameter's .grad now holds d(loss)/d(parameter) for this batch.
    loss.backward()
    # 4. Update. no_grad keeps the arithmetic out of the graph. The in-place subtraction keeps
    #    each Parameter object, and any object that holds it, pointing at the new values.
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.add_(p.grad, alpha=-lr)
    # 5. The loss describes the parameters that produced the gradient, so return it from before
    #    step 4. detach() and float() make it a plain number with no graph attached.
    return float(loss.detach())
