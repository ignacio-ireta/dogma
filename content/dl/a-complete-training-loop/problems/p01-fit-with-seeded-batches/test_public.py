from __future__ import annotations

import torch
from submission import fit


def tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(1, 1).double()


def tiny_data():
    X = torch.linspace(-1.0, 1.0, 6, dtype=torch.float64).reshape(6, 1)
    y = 2.0 * X + 0.5
    return X, y


def test_returns_one_train_and_one_validation_loss_per_epoch():
    X, y = tiny_data()
    train, val = fit(tiny_model(), X, y, X, y, epochs=3, batch_size=2, lr=0.1, seed=0)
    assert len(train) == 3 and len(val) == 3, "return one loss per epoch for each list"
    assert all(isinstance(v, float) for v in train + val), "losses must be Python floats"


def test_training_reduces_loss_on_a_linear_target():
    X, y = tiny_data()
    train, _ = fit(tiny_model(), X, y, X, y, epochs=30, batch_size=3, lr=0.2, seed=1)
    assert train[-1] < train[0], "a linear target should be fit: the loss must drop over epochs"


def test_single_epoch_and_batch_larger_than_data_is_one_batch():
    X, y = tiny_data()
    train, val = fit(tiny_model(), X, y, X, y, epochs=1, batch_size=100, lr=0.1, seed=2)
    assert len(train) == 1 and len(val) == 1, "one epoch must give exactly one entry per list"
