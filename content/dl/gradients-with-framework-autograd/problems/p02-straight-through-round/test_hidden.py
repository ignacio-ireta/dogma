from __future__ import annotations

import numpy as np
import torch
from submission import ste_quantize

ATOL = 1e-9  # matches meta.yaml tolerance
RTOL = 1e-9  # matches meta.yaml tolerance


def checked(call) -> torch.Tensor:
    try:
        out = call()
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: honor the documented contract"
        ) from None
    if not isinstance(out, torch.Tensor):
        raise AssertionError("return a torch.Tensor")
    return out


def rejects(call) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"invalid input must raise ValueError, not {type(exc).__name__}"
        ) from None
    raise AssertionError("invalid input was accepted: check the step and the input dtype")


def test_forward_matches_numpy_round_half_to_even_on_random_inputs():
    rng = np.random.default_rng(0)
    values = rng.normal(scale=3.0, size=(4, 5))
    step = 0.375
    expected = np.round(values / step) * step  # numpy also rounds half to even
    got = checked(lambda: ste_quantize(torch.tensor(values), step))
    assert got.dtype == torch.float64, "result dtype must match the float64 input"
    assert np.allclose(got.numpy(), expected, atol=ATOL, rtol=RTOL), (
        "forward values differ from round(x / step) * step"
    )


def test_ties_round_to_the_even_multiple_of_step():
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5], dtype=torch.float64)
    got = checked(lambda: ste_quantize(x, 1.0))
    assert torch.allclose(
        got, torch.tensor([0.0, 2.0, 2.0, -0.0, -2.0], dtype=torch.float64), atol=ATOL
    ), "ties must round half to even, as torch.round does"


def test_gradient_is_identity_for_every_entry_including_far_outside_any_range():
    x = torch.tensor([-5.0, 0.3, 7.9, 1e6], dtype=torch.float64, requires_grad=True)
    y = checked(lambda: ste_quantize(x, 0.5))
    weights = torch.tensor([1.0, -2.0, 0.5, 4.0], dtype=torch.float64)
    (y * weights).sum().backward()
    assert x.grad is not None, "output must stay connected to x for the backward pass"
    assert torch.allclose(x.grad, weights, atol=ATOL, rtol=RTOL), (
        "backward must be the identity: a clipped or zeroed gradient is the wrong estimator"
    )


def test_gradient_flows_through_a_downstream_nonlinearity_as_identity_times_its_slope():
    x = torch.tensor([0.1, 0.9, -1.2], dtype=torch.float64, requires_grad=True)
    y = checked(lambda: ste_quantize(x, 0.25))
    (y**2).sum().backward()
    assert torch.allclose(x.grad, 2 * y.detach(), atol=ATOL, rtol=RTOL), (
        "the upstream gradient 2*y must reach x unchanged through the quantizer"
    )


def test_staircase_rises_by_one_step_across_a_full_step_span_in_float64():
    # Q(x + s) - Q(x - s) = 2s for every point away from a tie, so the secant slope over one step
    # either side is 1. This checks the forward staircase, not the gradient.
    step = 0.5
    base = torch.tensor([0.07, 0.31, 1.13, -2.21], dtype=torch.float64)
    with torch.no_grad():
        up = ste_quantize(base + step, step)
        down = ste_quantize(base - step, step)
    secant = (up - down) / (2 * step)
    assert torch.allclose(secant, torch.ones_like(base), atol=ATOL, rtol=RTOL), (
        "across one step the quantizer should rise by exactly one step per step"
    )


def test_tiny_perturbations_do_not_change_the_forward_value_away_from_a_boundary():
    # The true derivative of the staircase is zero: nudging x a little changes nothing.
    # This is why the estimator has to replace the gradient, not compute it.
    x = torch.tensor([0.26, 0.74], dtype=torch.float64)
    h = 1e-7
    with torch.no_grad():
        plus = ste_quantize(x + h, 0.5)
        base = ste_quantize(x, 0.5)
    assert torch.allclose(plus, base, atol=ATOL, rtol=RTOL), (
        "the forward must be a staircase: a tiny nudge should not change the quantized value"
    )


def test_output_dtype_and_shape_follow_the_input_for_float32_and_zero_dim():
    x32 = torch.tensor([[0.1, 0.9], [1.6, -0.7]], dtype=torch.float32)
    out32 = checked(lambda: ste_quantize(x32, 0.5))
    assert out32.dtype == torch.float32 and out32.shape == (2, 2), (
        "output must keep the input dtype and shape"
    )
    scalar = checked(lambda: ste_quantize(torch.tensor(1.3, dtype=torch.float64), 0.5))
    assert scalar.dim() == 0 and abs(float(scalar) - 1.5) <= ATOL, (
        "a 0-dimensional input must work too"
    )


def test_huge_and_tiny_magnitudes_quantize_exactly():
    big = torch.tensor([1e12 + 0.4, -1e12 - 0.6], dtype=torch.float64)
    got_big = checked(lambda: ste_quantize(big, 1.0))
    expected_big = torch.tensor([1e12, -1e12 - 1.0], dtype=torch.float64)
    assert torch.allclose(got_big, expected_big, atol=0.0, rtol=RTOL), (
        "large magnitudes must round to the nearest integer multiple of step"
    )
    tiny = torch.tensor([3e-12, 7e-12], dtype=torch.float64)
    got_tiny = checked(lambda: ste_quantize(tiny, 1e-12))
    assert torch.allclose(
        got_tiny, torch.tensor([3e-12, 7e-12], dtype=torch.float64), atol=0.0, rtol=1e-9
    ), "tiny steps must keep the quantized scale"


def test_input_without_grad_gives_output_without_grad():
    y = checked(lambda: ste_quantize(torch.tensor([0.2, 0.7]), 0.5))
    assert not y.requires_grad, "a plain input must not produce a tracked output"


def test_input_tensor_values_are_not_mutated():
    x = torch.tensor([0.12, 0.88, -1.4], dtype=torch.float64)
    before = x.clone()
    checked(lambda: ste_quantize(x, 0.25))
    assert torch.equal(x, before), "the input values were modified: do not write into x"


def test_step_must_be_finite_and_positive_and_input_floating():
    x = torch.tensor([0.5, 1.0])
    rejects(lambda: ste_quantize(x, 0.0))
    rejects(lambda: ste_quantize(x, -0.5))
    rejects(lambda: ste_quantize(x, float("nan")))
    rejects(lambda: ste_quantize(x, float("inf")))
    rejects(lambda: ste_quantize(x, True))
    rejects(lambda: ste_quantize(torch.tensor([1, 2]), 0.5))
