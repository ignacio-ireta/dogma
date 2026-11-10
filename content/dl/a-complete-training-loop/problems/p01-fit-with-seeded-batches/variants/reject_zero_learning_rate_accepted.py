from __future__ import annotations

import copy
import math

import torch


def _check(X: torch.Tensor, y: torch.Tensor, name: str) -> None:
    if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[0] < 1:
        raise ValueError(f"{name}: inputs and targets must be 2-D with the same, nonzero row count")


def _int_in(value: int, low: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")


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
    """Wrong on purpose: a learning rate of zero is accepted.

    See problem.md for the contract. The shuffle comes from one generator seeded with seed, so
    the run does not depend on the global random state.
    """
    _check(X, y, "training")
    _check(X_val, y_val, "validation")
    _int_in(epochs, 1, "epochs")
    _int_in(batch_size, 1, "batch_size")
    _int_in(seed, 0, "seed")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
        raise ValueError("lr must be a finite number greater than zero")

    # The optimizer and the shuffle generator are fixed by the contract, so the run reproduces.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = math.inf
    best_state = None

    for _ in range(epochs):
        # One permutation per epoch; batches are consecutive slices of it, the last one partial.
        perm = torch.randperm(n, generator=generator)
        model.train()
        weighted_total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            prediction = model(X[idx])
            loss = ((prediction - y[idx]) ** 2).mean()
            # Clear, forward, loss, backward, step: the same five lines as the one-step topic.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss holds the value computed before step(), so this is the pre-update batch loss.
            # Weight by batch size so a partial last batch counts for its own examples only.
            weighted_total += loss.item() * idx.shape[0]
        train_losses.append(weighted_total / n)

        # Validation uses eval mode (dropout and batch-norm statistics frozen) and records no graph.
        model.eval()
        with torch.no_grad():
            val = float(((model(X_val) - y_val) ** 2).mean())
        val_losses.append(val)
        # Strictly lower only: a tie keeps the earlier, already-saved epoch.
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None  # epochs >= 1 guarantees at least one validation pass
    model.load_state_dict(best_state)
    model.eval()
    return train_losses, val_losses
