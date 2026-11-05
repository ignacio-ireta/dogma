# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math
from functools import partial

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only: the learner may not use torch.nn
from submission import lora_linear as _learner

ATOL = 1e-9
RTOL = 1e-9
FD_EPS = 1e-6
FD_TOL = 1e-6  # central differences in float64: truncation about 1e-12, rounding about 1e-10


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _rand(shape, seed, scale=1.0, dtype=torch.float64):
    return torch.randn(shape, generator=_gen(seed), dtype=dtype) * scale


def _call(*args, **kwargs):
    snapshot = [t.clone() if torch.is_tensor(t) else t for t in args]
    error = None
    try:
        out = _learner(*args, **kwargs)
    except ValueError as exc:
        error = f"valid arguments were rejected with ValueError ({exc}): check the shape rules"
    except Exception as exc:
        error = f"valid arguments raised {type(exc).__name__}: check shapes and the transposes"
    if error is not None:
        pytest.fail(error, pytrace=False)
    for before, after in zip(snapshot, args, strict=True):
        if torch.is_tensor(before) and not torch.equal(before, after):
            pytest.fail("an input tensor was modified in place; build new tensors", pytrace=False)
    return out


def _reference(x, W, b, A, B, alpha):  # noqa: PLR0917 (the contract fixes six arguments)
    """Base layer plus the scaled adapter, written with the library linear as the oracle."""
    r = A.shape[0]
    base = ref_f.linear(x, W, b)
    update = ref_f.linear(ref_f.linear(x, A), B)  # x A^T, then (.) B^T
    return base + (alpha / r) * update


def _expect_value_error(thunk, message):
    error = None
    try:
        thunk()
    except ValueError:
        return
    except Exception as exc:
        error = f"invalid input must raise ValueError, received {type(exc).__name__}"
    if error is None:
        error = message
    pytest.fail(error, pytrace=False)


def _case(seed, d_in, d_out, r, batch=(2, 3)):
    x = _rand((*batch, d_in), seed)
    W = _rand((d_out, d_in), seed + 1, scale=0.5)
    b = _rand((d_out,), seed + 2)
    A = _rand((r, d_in), seed + 3, scale=0.5)
    B = _rand((d_out, r), seed + 4, scale=0.5)
    return x, W, b, A, B


def test_unmerged_output_matches_the_library_reference_on_3d_inputs():
    for seed, (d_in, d_out, r) in enumerate([(4, 6, 2), (5, 3, 1), (3, 3, 3), (6, 2, 4)]):
        x, W, b, A, B = _case(seed * 10, d_in, d_out, r)
        out = _call(x, W, b, A, B, 1.5, merge=False)
        assert tuple(out.shape) == (2, 3, d_out), (
            "output must keep leading dims and become (..., d_out)"
        )
        torch.testing.assert_close(
            out,
            _reference(x, W, b, A, B, 1.5),
            atol=ATOL,
            rtol=RTOL,
            msg="unmerged output differs from the reference: check scale, transposes and bias",
        )


def test_merged_and_unmerged_outputs_agree_for_several_ranks():
    for seed, (d_in, d_out, r) in enumerate([(4, 6, 1), (4, 6, 2), (4, 6, 4), (7, 5, 3)]):
        x, W, b, A, B = _case(seed + 100, d_in, d_out, r)
        torch.testing.assert_close(
            _call(x, W, b, A, B, 0.75, merge=True),
            _call(x, W, b, A, B, 0.75, merge=False),
            atol=ATOL,
            rtol=RTOL,
            msg="merged and unmerged outputs differ: the merged weight must be W + (alpha / r) B A",
        )


def test_zero_b_gives_the_base_layer_for_any_input():
    x, W, b, A, _ = _case(7, 5, 4, 2)
    B0 = torch.zeros(4, 2, dtype=torch.float64)
    torch.testing.assert_close(
        _call(x, W, b, A, B0, 3.0),
        ref_f.linear(x, W, b),
        atol=1e-12,
        rtol=0,
        msg="a zero B must return the base layer exactly: the update must be a product with B",
    )


def test_scale_is_alpha_over_rank_not_alpha_or_inverse_rank():
    x, W, b, A, B = _case(21, 4, 4, 2)
    base = ref_f.linear(x, W, b)
    upd_r = _call(x, W, b, A, B, 2.0) - base  # alpha = r, so the scale is 1
    upd_big = _call(x, W, b, A, B, 6.0) - base  # alpha = 3 r, so the scale is 3
    torch.testing.assert_close(
        upd_big,
        3.0 * upd_r,
        atol=ATOL,
        rtol=RTOL,
        msg="the adapter term must scale linearly with alpha: check the scale is alpha / r",
    )
    torch.testing.assert_close(
        upd_r,
        ref_f.linear(ref_f.linear(x, A), B),
        atol=ATOL,
        rtol=RTOL,
        msg="with alpha equal to r the adapter term must be exactly x A^T B^T: the scale is wrong",
    )


def test_bias_is_added_once_and_not_scaled_by_alpha_over_rank():
    x, W, b, A, B = _case(31, 3, 3, 1)
    zero_W = torch.zeros_like(W)
    zero_A = torch.zeros_like(A)
    torch.testing.assert_close(
        _call(x, zero_W, b, zero_A, B, 5.0),
        b.expand_as(_call(x, zero_W, b, zero_A, B, 5.0)),
        atol=ATOL,
        rtol=RTOL,
        msg="the bias must be added once, unscaled: check where the bias enters the sum",
    )
    torch.testing.assert_close(
        _call(x, zero_W, b, zero_A, B, 5.0, merge=True),
        b.expand_as(_call(x, zero_W, b, zero_A, B, 5.0)),
        atol=ATOL,
        rtol=RTOL,
        msg="the merged path must add the bias once, unscaled",
    )


def test_bias_none_is_the_same_as_a_zero_bias():
    x, W, _, A, B = _case(41, 4, 3, 2)
    zero = torch.zeros(3, dtype=torch.float64)
    torch.testing.assert_close(
        _call(x, W, None, A, B, 1.0),
        _call(x, W, zero, A, B, 1.0),
        atol=ATOL,
        rtol=RTOL,
        msg="a None bias must behave as a zero bias",
    )


def _numeric_grad(f, t):
    """Central finite differences of the scalar f with respect to every entry of t (float64)."""
    grad = torch.zeros_like(t)
    flat, out = t.reshape(-1), grad.view(-1)
    for idx in range(flat.numel()):
        up, down = t.clone(), t.clone()
        up.view(-1)[idx] += FD_EPS
        down.view(-1)[idx] -= FD_EPS
        out[idx] = (f(up) - f(down)) / (2 * FD_EPS)
    return grad


def _loss_in_a(a_val, x, W, b, B0, G):  # noqa: PLR0917 (helper binds six inputs)
    return float((_reference(x, W, b, a_val, B0, 1.25) * G).sum())


def _loss_in_b(b_val, x, W, b, A0, G):  # noqa: PLR0917 (helper binds six inputs)
    return float((_reference(x, W, b, A0, b_val, 1.25) * G).sum())


def test_gradients_reach_a_and_b_and_match_finite_differences_on_both_paths():
    for merge in (False, True):
        x, W, b, A0, B0 = _case(51, 4, 3, 2, batch=(2,))
        G = _rand((2, 3), 53)  # random upstream gradient: the loss is a random linear form
        A = A0.clone().requires_grad_(True)
        B = B0.clone().requires_grad_(True)
        W_frozen = W.clone().requires_grad_(False)
        b_frozen = b.clone().requires_grad_(False)
        loss = (_call(x, W_frozen, b_frozen, A, B, 1.25, merge=merge) * G).sum()
        grad_A, grad_B = torch.autograd.grad(loss, [A, B])

        num_A = _numeric_grad(partial(_loss_in_a, x=x, W=W, b=b, B0=B0, G=G), A0)
        num_B = _numeric_grad(partial(_loss_in_b, x=x, W=W, b=b, A0=A0, G=G), B0)
        assert torch.allclose(grad_A, num_A, atol=FD_TOL, rtol=FD_TOL), (
            "gradient for A disagrees with finite differences: the adapter path must stay "
            "differentiable (no .detach(), no .data)"
        )
        assert torch.allclose(grad_B, num_B, atol=FD_TOL, rtol=FD_TOL), (
            "gradient for B disagrees with finite differences: the adapter path must stay "
            "differentiable (no .detach(), no .data)"
        )


def test_frozen_base_weight_and_bias_receive_no_gradient():
    x, W, b, A0, B0 = _case(61, 4, 3, 2, batch=(2,))
    A = A0.clone().requires_grad_(True)
    B = B0.clone().requires_grad_(True)
    W_frozen = W.clone().requires_grad_(False)
    b_frozen = b.clone().requires_grad_(False)
    out = _call(x, W_frozen, b_frozen, A, B, 2.0)
    out.sum().backward()
    assert W_frozen.grad is None, "the base weight received a gradient: it must stay frozen"
    assert b_frozen.grad is None, "the bias received a gradient: it must stay frozen"
    assert A.grad is not None and B.grad is not None, "A and B must receive gradients"


def test_merge_does_not_write_into_the_base_weight():
    x, W, b, A, B = _case(71, 4, 4, 2)
    before = W.clone()
    _call(x, W, b, A, B, 2.0, merge=True)
    assert torch.equal(W, before), (
        "merge wrote into W in place: build the merged weight as a new tensor"
    )


def test_large_alpha_with_small_factors_keeps_precision():
    x, W, b, A, B = _case(81, 4, 4, 2)
    A_small, B_small = A * 1e-6, B * 1e-6
    alpha = 1e6  # scale = 1e6 / 2, so the update is about 1e-12 times the factors
    torch.testing.assert_close(
        _call(x, W, b, A_small, B_small, alpha),
        _reference(x, W, b, A_small, B_small, alpha),
        atol=1e-9,
        rtol=1e-9,
        msg="large alpha with small factors lost precision: keep scale and product in float64",
    )


def test_small_alpha_with_large_factors_keeps_precision():
    x, W, b, A, B = _case(91, 4, 4, 2)
    A_big, B_big = A * 1e3, B * 1e3
    alpha = 1e-6
    torch.testing.assert_close(
        _call(x, W, b, A_big, B_big, alpha),
        _reference(x, W, b, A_big, B_big, alpha),
        atol=1e-9,
        rtol=1e-9,
        msg="small alpha with large factors differs: check the scale is applied to the update only",
    )


def test_output_dtype_follows_the_input_in_float32():
    x, W, b, A, B = _case(101, 4, 3, 2)
    to32 = [t.float() for t in (x, W, b, A, B)]
    out = _call(*to32[:3], to32[3], to32[4], 1.0)
    assert out.dtype == torch.float32, "the output must keep the dtype of x"
    # float32 rounding: a few ulps of values near 1 (about 1e-7 each), with headroom for 3 sums.
    torch.testing.assert_close(
        out,
        _reference(*(t.double() for t in (x, W, b, A, B)), 1.0).float(),
        atol=1e-5,
        rtol=1e-5,
        msg="float32 output differs from the reference beyond rounding: check the dtype path",
    )


def test_rank_one_and_full_rank_adapters_are_accepted():
    x, W, b, A, B = _case(111, 3, 3, 3)
    torch.testing.assert_close(
        _call(x, W, b, A, B, 2.0),
        _reference(x, W, b, A, B, 2.0),
        atol=ATOL,
        rtol=RTOL,
        msg="a full-rank adapter (r = d_in) must work: check the rank is read from A.shape[0]",
    )


def test_one_dimensional_input_gives_one_dimensional_output():
    x, W, b, A, B = _case(121, 4, 3, 2, batch=())
    out = _call(x, W, b, A, B, 1.0)
    assert tuple(out.shape) == (3,), "a single vector input must return shape (d_out,)"


@pytest.mark.parametrize(
    "case, why",
    [
        ("x_scalar", "a 0-dimensional x"),
        ("W_not_2d", "a W that is not 2-D"),
        ("x_width", "x whose width is not d_in"),
        ("A_wrong_width", "A whose width is not d_in"),
        ("A_empty", "an A with zero rows"),
        ("B_transposed", "a B given as (r, d_out) instead of (d_out, r)"),
        ("b_wrong", "a bias of the wrong length"),
        ("alpha_nan", "a NaN alpha"),
        ("alpha_inf", "an infinite alpha"),
        ("alpha_bool", "a bool alpha"),
        ("dtype_mix", "float32 A with float64 x"),
        ("int_x", "an integer x"),
    ],
    ids=[
        "x-scalar",
        "W-not-2d",
        "x-width",
        "A-wrong-width",
        "A-empty",
        "B-transposed",
        "bias-wrong",
        "alpha-nan",
        "alpha-inf",
        "alpha-bool",
        "dtype-mix",
        "int-x",
    ],
)
def test_invalid_arguments_raise_value_error(case, why):
    x, W, b, A, B = _case(131, 4, 3, 2)
    args = {"x": x, "W": W, "b": b, "A": A, "B": B, "alpha": 1.0}
    if case == "x_scalar":
        args["x"] = torch.tensor(1.0, dtype=torch.float64)
    elif case == "W_not_2d":
        args["W"] = W.reshape(-1)
    elif case == "x_width":
        args["x"] = _rand((2, 5), 9)
    elif case == "A_wrong_width":
        args["A"] = _rand((2, 5), 9)
    elif case == "A_empty":
        args["A"] = torch.zeros((0, 4), dtype=torch.float64)
        args["B"] = torch.zeros((3, 0), dtype=torch.float64)
    elif case == "B_transposed":
        args["B"] = B.T.contiguous()
    elif case == "b_wrong":
        args["b"] = torch.zeros(5, dtype=torch.float64)
    elif case == "alpha_nan":
        args["alpha"] = math.nan
    elif case == "alpha_inf":
        args["alpha"] = math.inf
    elif case == "alpha_bool":
        args["alpha"] = True
    elif case == "dtype_mix":
        args["A"] = A.float()
    elif case == "int_x":
        args["x"] = torch.ones((2, 4), dtype=torch.int64)
    _expect_value_error(
        lambda: _learner(**args),
        f"{why} was accepted instead of raising ValueError",
    )


def test_none_bias_and_zero_bias_both_accepted_in_merged_path():
    x, W, _, A, B = _case(141, 4, 3, 2)
    out = _call(x, W, None, A, B, 1.0, merge=True)
    assert tuple(out.shape) == (2, 3, 3), "the merged path must accept a None bias"


@pytest.mark.parametrize("merge", [False, True], ids=["unmerged", "merged"])
def test_integer_tensors_without_a_floating_dtype_raise_value_error(merge):
    x = torch.ones((2, 4), dtype=torch.int64)
    W = torch.ones((3, 4), dtype=torch.int64)
    A = torch.ones((2, 4), dtype=torch.int64)
    B = torch.ones((3, 2), dtype=torch.int64)
    _expect_value_error(
        lambda: _learner(x, W, None, A, B, 1.0, merge=merge),
        "tensors with no floating dtype must raise ValueError: x, W, A and B must be floating",
    )
