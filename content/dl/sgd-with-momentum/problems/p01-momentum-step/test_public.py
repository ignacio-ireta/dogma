from __future__ import annotations

import pytest
import torch
from submission import sgd_momentum_step


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_first_step_moves_by_learning_rate_times_gradient():
    params, velocity = sgd_momentum_step(
        [_tensor([1.0, 2.0])], [_tensor([0.5, -1.0])], None, lr=0.1, momentum=0.9
    )
    torch.testing.assert_close(
        params[0],
        _tensor([0.95, 2.1]),
        msg="first step must move by lr times the raw gradient (no momentum damping)",
    )
    torch.testing.assert_close(
        velocity[0],
        _tensor([0.5, -1.0]),
        msg="first step must seed the unscaled buffer with the raw gradient (lr only in update)",
    )


def test_second_step_accumulates_previous_direction():
    grads = [_tensor([0.5, -1.0])]
    params, velocity = sgd_momentum_step([_tensor([1.0, 2.0])], grads, None, 0.1, 0.9)
    params, velocity = sgd_momentum_step(params, grads, velocity, 0.1, 0.9)
    # buffer: 0.9 * [0.5, -1.0] + [0.5, -1.0] = [0.95, -1.9]
    torch.testing.assert_close(
        velocity[0],
        _tensor([0.95, -1.9]),
        msg="buffer must be momentum times the previous buffer plus the gradient",
    )
    torch.testing.assert_close(
        params[0],
        _tensor([0.855, 2.29]),
        msg="update must subtract lr times the buffer, not the raw gradient",
    )


def test_mismatched_parameter_and_gradient_counts_raise_value_error():
    with pytest.raises(ValueError):
        sgd_momentum_step([_tensor([1.0])], [], None, lr=0.1, momentum=0.9)
