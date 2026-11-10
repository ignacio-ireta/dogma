from __future__ import annotations

import copy
import math

import torch


def _is_int(v, minimum):
    return isinstance(v, int) and not isinstance(v, bool) and v >= minimum


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
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val must have the same number of rows")
    if not _is_int(epochs, 1):
        raise ValueError("epochs must be an int >= 1")
    if not _is_int(batch_size, 1):
        raise ValueError("batch_size must be an int >= 1")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite number greater than zero")
    if not _is_int(seed, 0):
        raise ValueError("seed must be an int >= 0")

    n = X.shape[0]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = math.inf
    best_state = None

    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        model.train()
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = ((model(X[idx]) - y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            val_loss = ((model(X_val) - y_val) ** 2).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return train_losses, val_losses
