# Public tests: the examples from problem.md plus one edge case.
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import residual_block


def test_identity_shortcut_example():
    """Identity shortcut: relu branch added to the input."""
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    out = residual_block(
        x,
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.allclose(out, torch.tensor([[2.0, -2.0]], dtype=torch.float64)), (
        "identity block output differs from the worked example"
    )


def test_projection_shortcut_example():
    """Projection shortcut: the linear map replaces the identity when widths differ."""
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    out = residual_block(
        x,
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        w_shortcut=torch.tensor([[0.5], [0.5]], dtype=torch.float64),
    )
    assert out.shape == (1, 1), "projection output must have width d_out"
    assert torch.allclose(out, torch.tensor([[0.5]], dtype=torch.float64)), (
        "projection shortcut output differs from the worked example"
    )


def test_identity_shortcut_rejects_width_change():
    """Edge case: an identity shortcut cannot change width, so it must raise ValueError."""
    x = torch.ones(1, 2, dtype=torch.float64)
    try:
        residual_block(
            x,
            torch.eye(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )
    except ValueError:
        return
    pytest.fail("identity shortcut with a width change must raise ValueError, not return a result")
