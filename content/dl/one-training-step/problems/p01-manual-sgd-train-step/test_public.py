from __future__ import annotations

import torch
from submission import train_step


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean()


def one_to_one_model(weight: float, bias: float) -> torch.nn.Linear:
    model = torch.nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.fill_(weight)
        model.bias.fill_(bias)
    return model


def test_one_step_updates_weight_and_bias_by_minus_lr_times_gradient():
    model = one_to_one_model(2.0, 0.0)
    train_step(
        model,
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
        mse,
        0.1,
    )
    assert abs(model.weight.item() - 1.6) < 1e-9, "weight must move by minus lr times its gradient"
    assert abs(model.bias.item() + 0.4) < 1e-9, "bias must move by minus lr times its gradient"


def test_returns_the_loss_from_before_the_update_as_a_float():
    model = one_to_one_model(2.0, 0.0)
    loss = train_step(
        model,
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
        mse,
        0.1,
    )
    assert isinstance(loss, float), "return a Python float"
    assert abs(loss - 4.0) < 1e-9, "return the loss at the parameters before the update"


def test_nonpositive_learning_rate_is_rejected_as_value_error():
    model = one_to_one_model(2.0, 0.0)
    x = torch.tensor([[1.0]], dtype=torch.float64)
    y = torch.tensor([[0.0]], dtype=torch.float64)
    try:
        train_step(model, x, y, mse, 0.0)
    except ValueError:
        return
    raise AssertionError("a learning rate of zero must raise ValueError")
