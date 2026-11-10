from __future__ import annotations

import pytest
import torch
from submission import sgd_momentum_step

SHAPES = [(3,), (2, 4), (1, 1, 3)]
LR = 0.1
MOMENTUM = 0.9


def _tensors(seed: int, scale: float = 1.0, shapes=SHAPES) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=gen, dtype=torch.float64) * scale for s in shapes]


def _grad_steps(seed: int, steps: int, scale: float = 1.0) -> list[list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed + 1000)
    return [
        [torch.randn(s, generator=gen, dtype=torch.float64) * scale for s in SHAPES]
        for _ in range(steps)
    ]


def _torch_reference(params0, grad_steps, lr, momentum):
    """torch.optim.SGD with dampening=0 and nesterov=False, traced after every step."""
    params = [p.clone().requires_grad_(True) for p in params0]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=0, nesterov=False)
    param_trace, buffer_trace = [], []
    for grads in grad_steps:
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        optimizer.step()
        param_trace.append([p.detach().clone() for p in params])
        buffer_trace.append(
            [
                None
                if optimizer.state[p].get("momentum_buffer") is None
                else optimizer.state[p]["momentum_buffer"].clone()
                for p in params
            ]
        )
    return param_trace, buffer_trace


def _submission_trace(params0, grad_steps, lr, momentum):
    params = [p.clone() for p in params0]
    velocity = None
    param_trace, buffer_trace = [], []
    for grads in grad_steps:
        try:
            params, velocity = sgd_momentum_step(
                params, [g.clone() for g in grads], velocity, lr, momentum
            )
        except Exception:
            pytest.fail(
                "a valid step raised an error: check the signature and return pair", pytrace=False
            )
        param_trace.append(params)
        buffer_trace.append(velocity)
    return param_trace, buffer_trace


def _assert_lists_match(actual, expected, message: str) -> None:
    for got, want in zip(actual, expected, strict=True):
        if got.shape != want.shape:
            pytest.fail(f"{message}: shape differs from the parameter shape", pytrace=False)
        if got.dtype != want.dtype:
            pytest.fail("output dtype must match the input parameter dtype", pytrace=False)
        if not torch.allclose(got, want, rtol=1e-10, atol=1e-10):
            pytest.fail(message, pytrace=False)


@pytest.mark.parametrize("seed", [0, 1, 2], ids=["run-a", "run-b", "run-c"])
def test_matches_torch_momentum_sgd_over_several_steps(seed):
    params0 = _tensors(seed)
    grad_steps = _grad_steps(seed, steps=6)
    expected_params, _ = _torch_reference(params0, grad_steps, LR, MOMENTUM)
    actual_params, _ = _submission_trace(params0, grad_steps, LR, MOMENTUM)
    for step, (got, want) in enumerate(zip(actual_params, expected_params, strict=True), 1):
        _assert_lists_match(
            got,
            want,
            f"step {step}: update differs from v = momentum * v + g, then p = p - lr * v",
        )


def test_first_step_seeds_buffer_with_raw_gradient_not_a_damped_one():
    params0 = _tensors(7)
    grads = _grad_steps(7, steps=1)[0]
    new_params, velocity = sgd_momentum_step(params0, grads, None, LR, MOMENTUM)
    _assert_lists_match(velocity, grads, "first step must seed the buffer with the raw gradient")
    _assert_lists_match(
        new_params,
        [p - LR * g for p, g in zip(params0, grads, strict=True)],
        "first step must move by lr times the raw gradient (no (1 - momentum) damping)",
    )


def test_zero_momentum_reduces_to_plain_gradient_descent():
    params0 = _tensors(11)
    grad_steps = _grad_steps(11, steps=4)
    expected_params, _ = _torch_reference(params0, grad_steps, LR, 0.0)
    actual_params, _ = _submission_trace(params0, grad_steps, LR, 0.0)
    for got, want in zip(actual_params, expected_params, strict=True):
        _assert_lists_match(got, want, "momentum=0 must reduce to plain gradient descent")


def test_velocity_state_is_the_unscaled_pytorch_buffer():
    params0 = _tensors(13)
    grad_steps = _grad_steps(13, steps=5)
    # lr != 1 so an lr-scaled buffer (Sutskever form) would disagree with torch's buffer.
    _, expected_buffers = _torch_reference(params0, grad_steps, 0.3, MOMENTUM)
    _, actual_buffers = _submission_trace(params0, grad_steps, 0.3, MOMENTUM)
    for got, want in zip(actual_buffers, expected_buffers, strict=True):
        _assert_lists_match(
            got,
            want,
            "velocity state must be the unscaled buffer: lr belongs only to the parameter update",
        )


def test_inputs_are_not_mutated():
    params = _tensors(17)
    grads = _grad_steps(17, steps=1)[0]
    velocity = _tensors(18)
    originals = [t.clone() for t in params + grads + velocity]
    sgd_momentum_step(params, grads, velocity, LR, MOMENTUM)
    for before, after in zip(originals, params + grads + velocity, strict=True):
        if not torch.equal(before, after):
            pytest.fail("input tensors were mutated: return new tensors instead", pytrace=False)


def _rejects(call, case: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(f"{case}: must raise ValueError, not another exception type", pytrace=False)
    pytest.fail(f"{case}: invalid input was accepted", pytrace=False)


@pytest.mark.parametrize(
    "case",
    [
        "parameter and gradient counts differ",
        "gradient shape differs from its parameter",
        "velocity count differs from parameter count",
        "negative learning rate",
        "non-finite learning rate",
        "momentum equal to one",
        "negative momentum",
    ],
    ids=[
        "parameter-and-gradient-counts-differ",
        "gradient-shape-differs-from-its-parameter",
        "velocity-count-differs-from-parameter-count",
        "negative-learning-rate",
        "non-finite-learning-rate",
        "momentum-equal-to-one",
        "negative-momentum",
    ],
)
def test_invalid_arguments_raise_value_error(case):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    calls = {
        "parameter and gradient counts differ": lambda: sgd_momentum_step(p, [], None, LR, 0.5),
        "gradient shape differs from its parameter": lambda: sgd_momentum_step(
            p, [torch.zeros(2, dtype=torch.float64)], None, LR, 0.5
        ),
        "velocity count differs from parameter count": lambda: sgd_momentum_step(p, g, [], LR, 0.5),
        "negative learning rate": lambda: sgd_momentum_step(p, g, None, -0.1, 0.5),
        "non-finite learning rate": lambda: sgd_momentum_step(p, g, None, float("nan"), 0.5),
        "momentum equal to one": lambda: sgd_momentum_step(p, g, None, LR, 1.0),
        "negative momentum": lambda: sgd_momentum_step(p, g, None, LR, -0.5),
    }
    _rejects(calls[case], case)


@pytest.mark.parametrize("scale", [1e-8, 1e8], ids=["tiny-gradients", "huge-gradients"])
def test_scale_extremes_match_torch(scale):
    params0 = _tensors(23, scale=scale)
    grad_steps = _grad_steps(23, steps=4, scale=scale)
    expected_params, _ = _torch_reference(params0, grad_steps, LR, MOMENTUM)
    actual_params, _ = _submission_trace(params0, grad_steps, LR, MOMENTUM)
    for got, want in zip(actual_params, expected_params, strict=True):
        _assert_lists_match(got, want, "update differs from torch at this magnitude scale")


def test_empty_parameter_list_returns_empty_lists():
    params, velocity = sgd_momentum_step([], [], None, LR, MOMENTUM)
    if list(params) != [] or list(velocity) != []:
        pytest.fail("an empty parameter list must return two empty lists", pytrace=False)


def test_float32_parameters_keep_their_dtype():
    p = [torch.ones(2, 2, dtype=torch.float32)]
    g = [torch.full((2, 2), 0.5, dtype=torch.float32)]
    new_params, velocity = sgd_momentum_step(p, g, None, LR, MOMENTUM)
    if new_params[0].dtype != torch.float32 or velocity[0].dtype != torch.float32:
        pytest.fail("output dtype must match the input parameter dtype", pytrace=False)


def test_zero_learning_rate_is_a_valid_step_that_keeps_parameters():
    params0 = _tensors(29)
    grads = _grad_steps(29, steps=1)[0]
    try:
        new_params, velocity = sgd_momentum_step(params0, grads, None, 0.0, MOMENTUM)
    except Exception as exc:
        pytest.fail(
            f"lr = 0 is allowed (lr >= 0), but a valid zero-rate step raised {type(exc).__name__}",
            pytrace=False,
        )
    _assert_lists_match(
        new_params, params0, "lr = 0 is allowed: the parameters must not move on a zero-rate step"
    )
    _assert_lists_match(
        velocity, grads, "lr = 0 is allowed: the buffer must still be the raw gradient on step one"
    )


def test_velocity_buffer_with_wrong_shape_raises_value_error():
    params = [torch.zeros(3, dtype=torch.float64)]
    grads = [torch.zeros(3, dtype=torch.float64)]
    bad_buffer = [torch.zeros(1, dtype=torch.float64)]
    try:
        sgd_momentum_step(params, grads, bad_buffer, LR, MOMENTUM)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "a velocity buffer of the wrong shape must raise ValueError: "
            f"raised {type(exc).__name__}",
            pytrace=False,
        )
    pytest.fail("a velocity buffer of the wrong shape must be rejected, not used", pytrace=False)


def test_returned_buffer_is_a_new_tensor_not_the_gradient():
    # The contract says "Return new tensors": editing a returned buffer must not touch the gradient.
    params = _tensors(31)
    grads = _grad_steps(31, steps=1)[0]
    kept = [g.clone() for g in grads]
    _, velocity = sgd_momentum_step(params, grads, None, LR, MOMENTUM)
    for buf in velocity:
        buf.add_(1.0)
    for got, want in zip(grads, kept, strict=True):
        if not torch.equal(got, want):
            pytest.fail(
                "the returned buffer must be a new tensor: editing it changed the gradient",
                pytrace=False,
            )


def test_infinite_learning_rate_is_rejected():
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    _rejects(lambda: sgd_momentum_step(p, g, None, float("inf"), 0.5), "infinite learning rate")
