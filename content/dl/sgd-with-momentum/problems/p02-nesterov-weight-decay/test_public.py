from __future__ import annotations

import pytest
import torch
from submission import sgd_nesterov_step


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_first_step_uses_decayed_gradient_with_nesterov_look_ahead():
    # d = 0.5 + 0.1 * 1.0 = 0.6, buffer = 0.6, look-ahead = 0.6 + 0.9 * 0.6 = 1.14
    params, velocity = sgd_nesterov_step(
        [_tensor([1.0])], [_tensor([0.5])], None, lr=0.1, momentum=0.9, weight_decay=0.1
    )
    torch.testing.assert_close(
        params[0],
        _tensor([0.886]),
        msg="first step must use the decayed gradient plus the momentum look-ahead",
    )
    torch.testing.assert_close(
        velocity[0],
        _tensor([0.6]),
        msg="first buffer must be the decayed gradient, not the raw gradient",
    )


def test_second_step_reuses_the_buffer_and_the_current_parameter():
    params, velocity = sgd_nesterov_step(
        [_tensor([1.0])], [_tensor([0.5])], None, lr=0.1, momentum=0.9, weight_decay=0.1
    )
    params, velocity = sgd_nesterov_step(
        params, [_tensor([0.5])], velocity, lr=0.1, momentum=0.9, weight_decay=0.1
    )
    torch.testing.assert_close(
        velocity[0],
        _tensor([1.1286]),
        msg="buffer must be momentum times the previous buffer plus the decayed gradient",
    )
    torch.testing.assert_close(
        params[0],
        _tensor([0.725566]),
        msg="update must use the current parameter and the look-ahead direction",
    )


def test_negative_weight_decay_raises_value_error():
    with pytest.raises(ValueError):
        sgd_nesterov_step(
            [_tensor([1.0])], [_tensor([0.5])], None, lr=0.1, momentum=0.9, weight_decay=-0.1
        )
