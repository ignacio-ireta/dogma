# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import swiglu_ffn


def _silu(z):
    return z / (1.0 + math.exp(-z))


def test_worked_example_gates_the_up_path_with_silu_of_the_gate():
    X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    V = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    W2 = torch.eye(2, dtype=torch.float64)
    want = torch.tensor([[_silu(1.0) * 3.0, _silu(2.0) * 2.0]], dtype=torch.float64)
    torch.testing.assert_close(
        swiglu_ffn(X, W, V, W2),
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="output must be W2 @ (SiLU(gate) * up): check which path is activated",
    )


def test_zero_input_gives_zero_output():
    X = torch.zeros(1, 2, dtype=torch.float64)
    W = torch.eye(2, dtype=torch.float64)
    V = torch.eye(2, dtype=torch.float64)
    out = swiglu_ffn(X, W, V, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(
        out,
        torch.zeros(1, 2, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
        msg="a zero input must give zero: SiLU(0) = 0 and the up path is 0",
    )


def test_gate_and_up_shapes_must_match():
    X = torch.zeros(1, 2, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        swiglu_ffn(X, eye, torch.zeros(3, 2, dtype=torch.float64), eye)
