from __future__ import annotations

import torch


def fit(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Train model with seeded mini-batch SGD; return per-epoch (train_losses, val_losses).

    Optimizer: torch.optim.SGD(model.parameters(), lr=lr), no momentum or decay. Loss:
    mean squared error over all elements of each batch. Shuffle: one g = torch.Generator()
    seeded with seed; each epoch draws perm = torch.randperm(n, generator=g) and batches are
    consecutive slices of perm (the last partial batch is kept). Train loss: mean of batch losses
    weighted by batch size, measured before each step. Validation: model.eval() and no_grad on all
    of X_val. Keep the state with the strictly lowest validation loss (earliest on ties) and load
    it into model at the end; call model.eval().
    Raises ValueError for mismatched sizes or out-of-range epochs, batch_size, lr or seed.
    Must not mutate any input. No torch.utils.data loaders or samplers.
    """
    raise NotImplementedError
