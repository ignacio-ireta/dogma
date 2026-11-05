# Public tests: the examples from problem.md plus the single-row edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import batch_norm_backward


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_two_row_example_has_zero_input_gradient():
    """With two rows the normalized values are always +-1, so grad_x vanishes."""
    gx, gg, gb = batch_norm_backward(_t([[1.0], [0.0]]), _t([[1.0], [3.0]]), _t([2.0]), eps=1e-12)
    assert gx.dtype == torch.float64, "grad_x must keep the dtype of x; do not downcast"
    assert torch.allclose(gx, _t([[0.0], [0.0]]), atol=1e-9), (
        "grad_x must vanish for two rows: check the mean and variance paths, and that the "
        "variance is the biased one (divide by N)"
    )
    assert torch.allclose(gg, _t([-1.0]), atol=1e-9), "grad_gamma must sum grad_y times x_hat"
    assert torch.allclose(gb, _t([1.0]), atol=1e-9), "grad_beta must sum grad_y over rows"


def test_constant_upstream_gradient_example():
    """A constant upstream gradient cannot move the normalized output, so grad_x is zero."""
    gx, gg, gb = batch_norm_backward(
        _t([[1.0], [1.0], [1.0]]), _t([[1.0], [3.0], [5.0]]), _t([2.0]), eps=1e-12
    )
    assert torch.allclose(gx, _t([[0.0], [0.0], [0.0]]), atol=1e-9), (
        "constant upstream gradient must give zero input gradient"
    )
    assert torch.allclose(gg, _t([0.0]), atol=1e-9), "grad_gamma must be zero here"
    assert torch.allclose(gb, _t([3.0]), atol=1e-9), "grad_beta must sum all three rows"


def test_single_row_is_rejected():
    """One row gives zero batch variance, so the backward pass must raise ValueError."""
    try:
        batch_norm_backward(_t([[1.0, 2.0]]), _t([[1.0, 2.0]]), _t([1.0, 1.0]))
    except ValueError:
        return
    raise AssertionError("one row must raise ValueError: the batch statistics are degenerate")
