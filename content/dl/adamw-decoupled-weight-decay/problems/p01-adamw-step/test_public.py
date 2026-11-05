from __future__ import annotations

import pytest
import torch
from submission import adamw_step


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_first_step_shrinks_weight_then_applies_adam_update():
    # shrink = 1 - 0.1 * 0.5 = 0.95 -> 0.95; Adam step with g = 0.5 is about lr = 0.1
    params, _ = adamw_step([_tensor([1.0])], [_tensor([0.5])], None, lr=0.1, weight_decay=0.5)
    torch.testing.assert_close(
        params[0],
        _tensor([0.85]),
        atol=1e-7,
        rtol=0,
        msg="AdamW must shrink the weight by (1 - lr * wd) and then take the Adam step",
    )


def test_zero_gradient_only_shrinks_the_weight_by_the_decay_factor():
    params, _ = adamw_step([_tensor([2.0])], [_tensor([0.0])], None, lr=0.1, weight_decay=0.5)
    torch.testing.assert_close(
        params[0],
        _tensor([1.9]),
        atol=1e-12,
        rtol=0,
        msg="zero gradient must shrink the weight by (1 - lr * wd), not by a sign-like step",
    )


def test_negative_weight_decay_raises_value_error():
    with pytest.raises(ValueError):
        adamw_step([_tensor([1.0])], [_tensor([0.5])], None, lr=0.1, weight_decay=-0.1)
