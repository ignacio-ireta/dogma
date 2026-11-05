# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import gelu_ffn


def _ident(n):
    return torch.eye(n, dtype=torch.float64)


def test_worked_example_applies_exact_gelu_between_the_two_layers():
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    zeros = torch.zeros(2, dtype=torch.float64)
    got = gelu_ffn(x, _ident(2), zeros, _ident(2), zeros)
    want = torch.tensor([[0.5 * (1 + math.erf(1 / math.sqrt(2))), 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="GELU(1) must be 1 * Phi(1) with the exact erf form: check the activation",
    )


def test_hidden_layer_bias_and_output_bias_are_both_added():
    x = torch.zeros(1, 2, dtype=torch.float64)
    b1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    W1 = torch.zeros(3, 2, dtype=torch.float64)
    W2 = torch.zeros(2, 3, dtype=torch.float64)
    b2 = torch.tensor([0.25, -0.5], dtype=torch.float64)
    got = gelu_ffn(x, W1, b1, W2, b2)
    torch.testing.assert_close(
        got,
        b2.expand(1, 2),
        atol=1e-9,
        rtol=0,
        msg="with zero weights the output must be the output bias: check that b2 is added",
    )


def test_wrong_output_weight_shape_raises_value_error():
    x = torch.zeros(1, 2, dtype=torch.float64)
    zeros = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        gelu_ffn(x, _ident(2), zeros, torch.zeros(2, 1, dtype=torch.float64), zeros)
