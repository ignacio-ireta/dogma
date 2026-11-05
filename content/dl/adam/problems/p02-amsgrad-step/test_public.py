from __future__ import annotations

import pytest
import torch
from submission import amsgrad_step


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_first_update_equals_adam_because_the_maximum_is_the_first_square():
    params, _ = amsgrad_step([_tensor([1.0])], [_tensor([2.0])], None, lr=0.1)
    torch.testing.assert_close(
        params[0],
        _tensor([0.9]),
        atol=1e-7,
        rtol=0,
        msg="first AMSGrad update must match the first Adam update",
    )


def test_a_small_gradient_after_a_large_one_keeps_the_large_denominator():
    params, state = amsgrad_step([_tensor([1.0])], [_tensor([2.0])], None, lr=0.1, beta2=0.5)
    params, state = amsgrad_step(params, [_tensor([0.1])], state, lr=0.1, beta2=0.5)
    torch.testing.assert_close(
        params[0],
        _tensor([0.8388]),
        atol=1e-4,
        rtol=0,
        msg="denominator must use the running maximum of the second moment",
    )


def test_beta1_equal_to_one_raises_value_error():
    try:
        amsgrad_step([_tensor([1.0])], [_tensor([2.0])], None, lr=0.1, beta1=1.0)
    except ValueError:
        return
    pytest.fail("beta1 = 1 must be rejected with ValueError", pytrace=False)
