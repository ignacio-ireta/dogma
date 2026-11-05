from __future__ import annotations

import torch
from submission import ste_quantize


def test_forward_rounds_each_entry_to_the_nearest_step_multiple():
    out = ste_quantize(torch.tensor([0.4, 0.6, 1.3]), 0.5)
    assert torch.allclose(out, torch.tensor([0.5, 0.5, 1.5]), atol=1e-9), (
        "forward must snap each value to the nearest multiple of step"
    )


def test_gradient_is_identity_even_though_the_forward_is_a_staircase():
    x = torch.tensor([0.2, 1.9], requires_grad=True)
    y = ste_quantize(x, 1.0)
    assert torch.allclose(y.detach(), torch.tensor([0.0, 2.0]), atol=1e-9), (
        "forward values are wrong: check round(x / step) * step"
    )
    (3 * y).sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.tensor([3.0, 3.0])), (
        "gradient must pass straight through: the round itself must not block it"
    )


def test_input_without_grad_gives_output_without_grad():
    y = ste_quantize(torch.tensor([0.3, 0.8]), 0.5)
    assert not y.requires_grad, "an input that does not track gradients must not give one"
