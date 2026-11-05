# Hidden tests: autograd and finite-difference oracles on random batches, plus edge cases.
# Autograd and finite differences are used only here; the submission must derive gradients in
# closed form. Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import batch_norm_backward

ATOL = 1e-9
RTOL = 1e-9
# Central differences with h = 1e-6 in float64 are accurate to about 1e-8 for these magnitudes;
# the finite-difference comparison uses 1e-6, looser than the analytic ones, for that reason.
FD_TOL = 1e-6
FD_STEP = 1e-6


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=dtype) * scale + offset


def _call(*args, **kwargs):
    snapshots = [a.clone() for a in args if torch.is_tensor(a)]
    try:
        out = batch_norm_backward(*args, **kwargs)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, dtypes and the contract",
            pytrace=False,
        )
    for before, after in zip(snapshots, [a for a in args if torch.is_tensor(a)], strict=True):
        if not torch.equal(before, after):
            pytest.fail("an input tensor was modified in place; return new tensors", pytrace=False)
    is_triple = isinstance(out, tuple) and len(out) == 3 and all(torch.is_tensor(t) for t in out)
    if is_triple and any(t.dtype != args[1].dtype for t in out):
        pytest.fail("gradients must keep the dtype of x; do not downcast", pytrace=False)
    return out


def _result(out, n, c):
    if not isinstance(out, tuple | list) or len(out) != 3:
        pytest.fail("return three gradients: grad_x, grad_gamma and grad_beta", pytrace=False)
    if not all(torch.is_tensor(t) for t in out):
        pytest.fail("return torch tensors, not NumPy arrays or Python lists", pytrace=False)
    gx, gg, gb = out
    if tuple(gx.shape) != (n, c):
        pytest.fail("grad_x must have the input shape (N, C)", pytrace=False)
    if tuple(gg.shape) != (c,) or tuple(gb.shape) != (c,):
        pytest.fail("grad_gamma and grad_beta must have shape (C,)", pytrace=False)
    return gx, gg, gb


def _close(actual, expected, message, atol=ATOL, rtol=RTOL):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        pytest.fail(message, pytrace=False)


def _autograd(x, grad_y, gamma, eps):
    """Reference gradients from autograd through PyTorch's training-mode batch norm."""
    x_ = x.clone().requires_grad_(True)
    g_ = gamma.clone().requires_grad_(True)
    b_ = torch.zeros_like(gamma).requires_grad_(True)
    y = ref_f.batch_norm(x_, None, None, g_, b_, training=True, eps=eps)
    return torch.autograd.grad((y * grad_y).sum(), [x_, g_, b_])


def _objective(x, gamma, beta, grad_y, eps):
    return (ref_f.batch_norm(x, None, None, gamma, beta, training=True, eps=eps) * grad_y).sum()


def _finite_difference(variable, fn):
    numeric = torch.zeros_like(variable)
    for index in range(variable.numel()):
        flat = variable.reshape(-1)
        old = flat[index].item()
        flat[index] = old + FD_STEP
        plus = fn()
        flat[index] = old - FD_STEP
        minus = fn()
        flat[index] = old
        numeric.reshape(-1)[index] = (plus - minus) / (2 * FD_STEP)
    return numeric


SHAPES = [(2, 3), (3, 1), (4, 3), (5, 2), (8, 4)]


def test_gradients_match_autograd_on_random_batches():
    for seed, (n, c) in enumerate(SHAPES):
        x = _randn((n, c), seed, scale=2.0, offset=0.3)
        gamma = _randn((c,), seed + 10)
        grad_y = _randn((n, c), seed + 20)
        gx, gg, gb = _result(_call(grad_y, x, gamma, eps=1e-5), n, c)
        want_gx, want_gg, want_gb = _autograd(x, grad_y, gamma, 1e-5)
        _close(
            gx,
            want_gx,
            "grad_x differs from autograd: check the mean and variance paths, the gamma scaling, "
            "the biased variance (divide by N), and eps inside the square root",
        )
        _close(
            gg, want_gg, "grad_gamma differs from autograd: it must sum grad_y * x_hat over rows"
        )
        _close(gb, want_gb, "grad_beta differs from autograd: it must sum grad_y over rows")


def test_gradients_match_finite_differences_in_float64():
    x = _randn((4, 3), 31)
    gamma = _randn((3,), 32)
    beta = _randn((3,), 33)
    grad_y = _randn((4, 3), 34)
    eps = 1e-3

    def objective():
        return float(_objective(x, gamma, beta, grad_y, eps))

    gx, gg, gb = _result(_call(grad_y, x, gamma, eps=eps), 4, 3)
    numeric = {
        "x": _finite_difference(x, objective),
        "gamma": _finite_difference(gamma, objective),
        "beta": _finite_difference(beta, objective),
    }
    _close(
        gx,
        numeric["x"],
        "grad_x disagrees with finite differences: check the mean and variance paths "
        "and the gamma scaling",
        atol=FD_TOL,
        rtol=FD_TOL,
    )
    _close(
        gg,
        numeric["gamma"],
        "grad_gamma disagrees with finite differences",
        atol=FD_TOL,
        rtol=FD_TOL,
    )
    _close(
        gb, numeric["beta"], "grad_beta disagrees with finite differences", atol=FD_TOL, rtol=FD_TOL
    )


def test_constant_upstream_gradient_gives_zero_input_gradient():
    x = _randn((5, 2), 41)
    gamma = _randn((2,), 42)
    grad_y = torch.full((5, 2), 0.75, dtype=torch.float64)
    gx, _, gb = _result(_call(grad_y, x, gamma, eps=1e-5), 5, 2)
    _close(
        gx,
        torch.zeros_like(gx),
        "constant upstream gradient must give zero grad_x: the mean or variance path is wrong "
        "(a constant grad_y cannot move the normalized output)",
    )
    _close(gb, torch.full((2,), 3.75, dtype=torch.float64), "grad_beta must sum over the rows")


def test_varying_upstream_gradient_includes_variance_path():
    # A non-constant upstream gradient exercises the x_hat term, which vanishes for constant grad_y.
    x = _randn((6, 4), 51, scale=3.0)
    gamma = _randn((4,), 52)
    grad_y = _randn((6, 4), 53)
    gx, _, _ = _result(_call(grad_y, x, gamma, eps=1e-5), 6, 4)
    want_gx, _, _ = _autograd(x, grad_y, gamma, 1e-5)
    _close(gx, want_gx, "variance path of grad_x is wrong: the x_hat term is missing or mis-scaled")


def test_backward_uses_biased_variance_like_the_forward_pass():
    # With N=3 the unbiased variance differs from the biased one by 50 percent.
    x = _randn((3, 2), 61)
    gamma = torch.ones(2, dtype=torch.float64)
    grad_y = _randn((3, 2), 62)
    gx, _, _ = _result(_call(grad_y, x, gamma, eps=1e-5), 3, 2)
    want_gx, _, _ = _autograd(x, grad_y, gamma, 1e-5)
    _close(
        gx,
        want_gx,
        "backward used unbiased variance; it must differentiate the biased-variance forward pass",
    )


def test_parameter_gradients_sum_over_rows_not_average():
    x = _randn((4, 3), 71)
    gamma = _randn((3,), 72)
    grad_y = _randn((4, 3), 73)
    _, gg, gb = _result(_call(grad_y, x, gamma, eps=1e-5), 4, 3)
    _close(gb, grad_y.sum(dim=0), "grad_beta must be the sum over rows, not a mean")
    centred = x - x.mean(dim=0)
    x_hat = centred / torch.sqrt((centred * centred).mean(dim=0) + 1e-5)
    _close(gg, (grad_y * x_hat).sum(dim=0), "grad_gamma must be the sum over rows, not a mean")


def test_large_common_offset_matches_autograd():
    x = _randn((5, 2), 81, scale=1.0, offset=1e6)
    gamma = _randn((2,), 82)
    grad_y = _randn((5, 2), 83)
    gx, gg, gb = _result(_call(grad_y, x, gamma, eps=1e-5), 5, 2)
    want_gx, want_gg, want_gb = _autograd(x, grad_y, gamma, 1e-5)
    _close(
        gx,
        want_gx,
        "large offset destroyed grad_x: centre the inputs before squaring",
        atol=1e-7,
        rtol=1e-7,
    )
    _close(
        gg,
        want_gg,
        "large offset destroyed grad_gamma: centre the inputs before squaring",
        atol=1e-7,
        rtol=1e-7,
    )
    _close(gb, want_gb, "large offset destroyed grad_beta", atol=1e-7, rtol=1e-7)


def test_float32_inputs_return_float32_gradients():
    x = _randn((4, 3), 91, dtype=torch.float32)
    gamma = torch.ones(3, dtype=torch.float32)
    grad_y = _randn((4, 3), 92, dtype=torch.float32)
    gx, gg, gb = _result(_call(grad_y, x, gamma, eps=1e-5), 4, 3)
    assert gx.dtype == torch.float32 and gg.dtype == torch.float32 and gb.dtype == torch.float32, (
        "gradients must keep the input dtype"
    )
    want_gx, want_gg, want_gb = _autograd(x, grad_y, gamma, 1e-5)
    _close(gx, want_gx, "float32 grad_x differs from autograd", atol=1e-4, rtol=1e-4)
    _close(gg, want_gg, "float32 grad_gamma differs from autograd", atol=1e-4, rtol=1e-4)
    _close(gb, want_gb, "float32 grad_beta differs from autograd", atol=1e-4, rtol=1e-4)


def test_inputs_are_not_mutated():
    x = _randn((4, 3), 101)
    gamma = _randn((3,), 102)
    grad_y = _randn((4, 3), 103)
    originals = [t.clone() for t in (grad_y, x, gamma)]
    _call(grad_y, x, gamma, eps=1e-5)
    for name, now, before in zip(
        ("grad_y", "x", "gamma"), (grad_y, x, gamma), originals, strict=True
    ):
        assert torch.equal(now, before), f"input {name} was modified in place; return new tensors"


def test_invalid_arguments_raise_value_error():
    x = _randn((4, 3), 111)
    g = torch.ones(3, dtype=torch.float64)
    gy = _randn((4, 3), 112)
    cases = {
        "a one-dimensional x must raise ValueError": lambda: batch_norm_backward(gy, x[0], g),
        "an integer x must raise ValueError": lambda: batch_norm_backward(
            gy, torch.ones(4, 3, dtype=torch.int64), g
        ),
        "grad_y with a different shape must raise ValueError": lambda: batch_norm_backward(
            gy[:3], x, g
        ),
        "gamma with the wrong feature count must raise ValueError": lambda: batch_norm_backward(
            gy, x, torch.ones(2, dtype=torch.float64)
        ),
        "a zero eps must raise ValueError": lambda: batch_norm_backward(gy, x, g, eps=0.0),
        "a single row must raise ValueError": lambda: batch_norm_backward(gy[:1], x[:1], g),
    }
    for message, thunk in cases.items():
        try:
            thunk()
        except ValueError:
            continue
        except Exception:
            pytest.fail(
                "invalid input must raise ValueError, not another exception type", pytrace=False
            )
        pytest.fail(message, pytrace=False)


def test_default_eps_matches_the_documented_signature():
    # Small variance (about 1e-5) so that a different default eps changes the gradients clearly.
    x = _randn((6, 3), 201, scale=3e-3)
    gamma = _randn((3,), 202)
    grad_y = _randn((6, 3), 203)
    gx, gg, gb = _result(_call(grad_y, x, gamma), 6, 3)
    want_gx, want_gg, want_gb = _autograd(x, grad_y, gamma, 1e-5)
    _close(
        gx,
        want_gx,
        "default eps differs from the documented signature (eps 1e-5): grad_x is off",
    )
    _close(
        gg,
        want_gg,
        "default eps differs from the documented signature (eps 1e-5): grad_gamma is off",
    )
    _close(
        gb,
        want_gb,
        "default eps differs from the documented signature (eps 1e-5): grad_beta is off",
    )
