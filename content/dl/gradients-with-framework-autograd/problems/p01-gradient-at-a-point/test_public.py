from __future__ import annotations

import torch
from submission import grad_at


def test_gradient_of_sum_of_squares_is_twice_the_point():
    x = torch.tensor([1.0, 2.0, 3.0])
    out = grad_at(lambda v: (v**2).sum(), x)
    assert isinstance(out, torch.Tensor), "return one gradient tensor, not a tuple"
    assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0])), (
        "gradient of sum(x**2) is 2x: check the chain rule through the power"
    )


def test_caller_tensor_keeps_requires_grad_flag_and_no_grad_field():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    g = grad_at(lambda v: (v**3).sum(), x)
    assert isinstance(g, torch.Tensor), "return one gradient tensor, not a tuple"
    assert torch.allclose(g, torch.tensor([3.0, 12.0], dtype=torch.float64)), (
        "gradient of sum(x**3) is 3x**2"
    )
    assert x.requires_grad, "the caller's requires_grad flag was changed"
    assert x.grad is None, "the function wrote into the caller's .grad field"


def test_constant_function_gives_zero_gradient_not_an_error():
    x = torch.tensor([0.5, -1.5])
    g = grad_at(lambda v: torch.tensor(4.0), x)
    assert torch.equal(g, torch.zeros(2)), "a constant has zero derivative with respect to x"
