from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import forward_backward

"""Hidden tests for forward_backward.

Two independent oracles: the hand-derived closed form (dL/dz = 2r(w + e)(1 - h^2), then the
paths into each input) and central differences of a separate copy of the forward graph.
Every failure message names the kind of mistake and never shows a hidden input or value.
"""


KEYS = ("x", "w", "b", "t")
VALUE = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
REL_TOL = 1e-9
FD_TOL = 1e-6
FD_STEP = 1e-6

KEY_MESSAGE = {
    "x": "dL/dx is wrong: check the local derivative of p = w*x with respect to x",
    "w": "dL/dw is wrong: w feeds p = w*x and q = h*w, so both contributions must be added",
    "b": "dL/db is wrong: z = p + b passes its gradient to b with local derivative 1",
    "t": "dL/dt is wrong: r = y - t has local derivative -1 with respect to t",
}


def _loss(x, w, b, t):
    """The graph written as one expression, separate from the solution's structure."""
    h = math.tanh(w * x + b)
    y = h * w + math.exp(h)
    r = y - t
    return r * r


def _closed_form(x, w, b, t):
    """Hand-derived gradients: dL/dz = 2r (w + exp(h)) (1 - h^2), then one term per path."""
    h = math.tanh(w * x + b)
    e = math.exp(h)
    r = h * w + e - t
    dz = 2.0 * r * (w + e) * (1.0 - h * h)
    return r * r, {"x": dz * w, "w": 2.0 * r * h + dz * x, "b": dz, "t": -2.0 * r}


def _close(got, want, tol=REL_TOL):
    return abs(got - want) <= tol * max(1.0, abs(want))


def _run(x, w, b, t):
    try:
        return forward_backward(x, w, b, t)
    except Exception as exc:
        pytest.fail(
            f"forward_backward raised {type(exc).__name__} on valid finite inputs: "
            "check the operations and the return value"
        )


def _checked(result):
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("return (loss, grads): a tuple of exactly two items")
    loss, grads = result
    if not isinstance(grads, dict) or set(grads) != set(KEYS):
        pytest.fail("grads must be a dict with exactly the keys x, w, b and t")
    for name, value in (("loss", loss), *grads.items()):
        if type(value) is not float:
            pytest.fail(f"{name} must be a plain Python float")
    return loss, grads


def test_loss_matches_the_stated_graph_on_seeded_random_inputs():
    rng = random.Random(0)  # noqa: S311 — deterministic numeric test data
    for _ in range(200):
        args = [rng.uniform(-2.0, 2.0) for _ in range(4)]
        loss, _ = _checked(_run(*args))
        assert _close(loss, _loss(*args)), (
            "loss is wrong: the forward pass must compute h = tanh(w*x + b), "
            "y = h*w + exp(h), then L = (y - t)*(y - t)"
        )


def test_gradients_match_closed_form_on_seeded_random_inputs():
    rng = random.Random(1)  # noqa: S311 — deterministic numeric test data
    for _ in range(200):
        args = [rng.uniform(-2.0, 2.0) for _ in range(4)]
        _, grads = _checked(_run(*args))
        _, want = _closed_form(*args)
        for key in KEYS:
            assert _close(grads[key], want[key]), KEY_MESSAGE[key]


def test_gradients_match_central_differences_of_the_forward_pass():
    rng = random.Random(2)  # noqa: S311 — deterministic numeric test data
    for _ in range(100):
        args = [rng.uniform(-2.0, 2.0) for _ in range(4)]
        _, grads = _checked(_run(*args))
        for i, key in enumerate(KEYS):
            plus, minus = list(args), list(args)
            plus[i] += FD_STEP
            minus[i] -= FD_STEP
            numeric = (_loss(*plus) - _loss(*minus)) / (2.0 * FD_STEP)
            assert _close(grads[key], numeric, tol=FD_TOL), (
                f"dL/d{key} disagrees with central differences of the forward pass: "
                "check the backward pass for that input"
            )


@pytest.mark.parametrize(
    "args",
    [(0.7, -1.2, 0.4, 0.9), (-1.5, 0.3, -0.2, 0.1), (2.0, 2.0, 0.0, -1.0)],
    ids=["mixed-signs", "small-weight", "large-residual"],
)
def test_w_gradient_adds_the_path_through_q_and_the_path_through_p(args):
    _, grads = _checked(_run(*args))
    _, want = _closed_form(*args)
    assert _close(grads["w"], want["w"]), (
        "dL/dw is missing a path: w feeds both p = w*x and q = h*w, so add both "
        "contributions (accumulate with +=, do not overwrite)"
    )


def test_zero_residual_gives_zero_loss_and_zero_gradients():
    x, w, b = 0.8, -0.6, 0.3
    h = math.tanh(w * x + b)
    t = h * w + math.exp(h)  # the target equals the model output, so r = 0
    loss, grads = _checked(_run(x, w, b, t))
    assert abs(loss) <= 1e-12, "zero residual must give zero loss: check r = y - t"
    for key in KEYS:
        assert abs(grads[key]) <= 1e-9, (
            "with zero residual every gradient must vanish, because each path carries "
            "the residual r; check the factor 2r from r*r and the factor from y - t"
        )


@pytest.mark.parametrize(
    "args",
    [(40.0, 1.0, 0.0, 0.5), (-400.0, 2.0, 0.0, -0.5)],
    ids=["saturated-positive", "saturated-negative"],
)
def test_saturated_tanh_gives_finite_gradients_matching_closed_form(args):
    _, grads = _checked(_run(*args))
    _, want = _closed_form(*args)
    for key in KEYS:
        assert math.isfinite(grads[key]), (
            "saturated tanh must give finite gradients: use 1 - h*h with h = tanh(z), "
            "not a formula that overflows for large |z| such as 1/cosh(z)**2"
        )
        assert _close(grads[key], want[key]), KEY_MESSAGE[key]


def test_repeated_calls_do_not_share_state():
    first = _checked(_run(0.5, -0.4, 0.2, 0.1))
    _checked(_run(-1.0, 1.5, 0.0, 0.3))
    again = _checked(_run(0.5, -0.4, 0.2, 0.1))
    assert again == first, (
        "results depend on earlier calls: keep no accumulators outside the function body"
    )


def test_integer_inputs_are_accepted_and_match_float_inputs():
    as_int = _checked(_run(1, 2, 0, 3))
    as_float = _checked(_run(1.0, 2.0, 0.0, 3.0))
    assert _close(as_int[0], as_float[0]), "integer inputs must give the same loss as floats"
    for key in KEYS:
        assert _close(as_int[1][key], as_float[1][key]), (
            "integer inputs must give the same gradients as floats: convert before arithmetic"
        )


def test_return_value_is_loss_and_a_dict_with_exactly_four_keys():
    result = _checked(_run(0.3, -0.8, 0.1, 0.2))
    assert isinstance(result, tuple) and len(result) == 2, "return a (loss, grads) tuple"
    loss, grads = result
    assert isinstance(loss, float), "loss must be a float"
    assert isinstance(grads, dict) and set(grads) == set(KEYS), (
        "grads must be a dict keyed exactly by x, w, b and t"
    )
    assert all(isinstance(v, float) for v in grads.values()), "every gradient must be a float"


@pytest.mark.parametrize("position", range(4), ids=["nan-x", "nan-w", "nan-b", "nan-t"])
def test_nan_input_raises_value_error(position):
    _raises_value_error(position, float("nan"))


@pytest.mark.parametrize("position", range(4), ids=["inf-x", "inf-w", "inf-b", "inf-t"])
def test_infinite_input_raises_value_error(position):
    _raises_value_error(position, float("inf"))


def _raises_value_error(position, bad):
    args = [0.5, -0.5, 0.25, 0.75]
    args[position] = bad
    try:
        forward_backward(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"non-finite input raised {type(exc).__name__} instead of ValueError: "
            "check every input with a finiteness test before the forward pass"
        )
    pytest.fail(
        "a non-finite input was accepted: check every input with a finiteness test "
        "before the forward pass"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(VALUE, VALUE, VALUE, VALUE)
def test_random_inputs_match_closed_form(x, w, b, t):
    _, grads = _checked(_run(x, w, b, t))
    _, want = _closed_form(x, w, b, t)
    for key in KEYS:
        assert _close(grads[key], want[key]), KEY_MESSAGE[key]


def test_nonzero_integer_bias_gives_the_same_result_as_float_bias():
    # An int bias of 1 must act as 1.0 in z = p + b. Dropping a non-zero int bias changes z.
    as_int = _checked(_run(1.0, 0.5, 1, 0.0))
    as_float = _checked(_run(1.0, 0.5, 1.0, 0.0))
    assert _close(as_int[0], as_float[0]), (
        "the loss with an integer bias must equal the loss with the same bias as a float"
    )
    for key in KEYS:
        assert _close(as_int[1][key], as_float[1][key]), (
            "gradients with an integer bias must equal those with the float bias: "
            "convert every input before arithmetic"
        )
