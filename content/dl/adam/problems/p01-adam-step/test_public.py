from __future__ import annotations

import pytest
import torch
from submission import adam_step


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("gradient", [2.0, 200.0], ids=["small-gradient", "large-gradient"])
def test_first_step_moves_about_learning_rate_whatever_the_gradient_scale(gradient):
    params, _ = adam_step([_tensor([1.0])], [_tensor([gradient])], None, lr=0.1)
    torch.testing.assert_close(
        params[0],
        _tensor([0.9]),
        atol=1e-7,
        rtol=0,
        msg="first Adam step must be about lr in size: check bias correction and eps placement",
    )


def test_constant_gradient_gives_equal_steps_after_bias_correction():
    params, state = adam_step([_tensor([1.0])], [_tensor([2.0])], None, lr=0.1)
    params, state = adam_step(params, [_tensor([2.0])], state, lr=0.1)
    torch.testing.assert_close(
        params[0],
        _tensor([0.8]),
        atol=1e-7,
        rtol=0,
        msg="second step must also be about lr in size for a constant gradient",
    )
    if state[2] != 2:
        pytest.fail("the step counter must be 2 after two updates", pytrace=False)


def test_beta_equal_to_one_raises_value_error():
    try:
        adam_step([_tensor([1.0])], [_tensor([2.0])], None, lr=0.1, beta1=1.0)
    except ValueError:
        return
    pytest.fail("beta1 = 1 must be rejected: the first moment would never forget", pytrace=False)
