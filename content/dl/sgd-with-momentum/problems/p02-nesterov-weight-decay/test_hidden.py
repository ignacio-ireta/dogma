from __future__ import annotations

import pytest
import torch
from submission import sgd_nesterov_step

SHAPES = [(3,), (2, 4), (1, 1, 3)]
LR = 0.1
MOMENTUM = 0.9
DECAY = 0.05


def _tensors(seed: int, scale: float = 1.0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=gen, dtype=torch.float64) * scale for s in SHAPES]


def _grad_steps(seed: int, steps: int, scale: float = 1.0) -> list[list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed + 1000)
    return [
        [torch.randn(s, generator=gen, dtype=torch.float64) * scale for s in SHAPES]
        for _ in range(steps)
    ]


def _torch_reference(params0, grad_steps, lr, momentum, weight_decay):
    """torch.optim.SGD with Nesterov and coupled weight decay, traced after every step."""
    params = [p.clone().requires_grad_(True) for p in params0]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, dampening=0, nesterov=True, weight_decay=weight_decay
    )
    param_trace, buffer_trace = [], []
    for grads in grad_steps:
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        optimizer.step()
        param_trace.append([p.detach().clone() for p in params])
        buffer_trace.append([optimizer.state[p]["momentum_buffer"].clone() for p in params])
    return param_trace, buffer_trace


def _submission_trace(params0, grad_steps, lr, momentum, weight_decay):
    params = [p.clone() for p in params0]
    velocity = None
    param_trace, buffer_trace = [], []
    for grads in grad_steps:
        try:
            params, velocity = sgd_nesterov_step(
                params,
                [g.clone() for g in grads],
                velocity,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        except Exception:
            pytest.fail(
                "a valid step raised an error: check the signature and return pair",
                pytrace=False,
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
def test_matches_torch_nesterov_sgd_with_weight_decay_over_steps(seed):
    params0 = _tensors(seed)
    grad_steps = _grad_steps(seed, steps=6)
    expected, _ = _torch_reference(params0, grad_steps, LR, MOMENTUM, DECAY)
    actual, _ = _submission_trace(params0, grad_steps, LR, MOMENTUM, DECAY)
    for step, (got, want) in enumerate(zip(actual, expected, strict=True), 1):
        _assert_lists_match(
            got,
            want,
            f"step {step}: Nesterov update differs from torch.optim.SGD(nesterov=True)",
        )


def test_weight_decay_joins_gradient_before_momentum():
    params0 = _tensors(5)
    grads = _grad_steps(5, steps=1)[0]
    new_params, velocity = sgd_nesterov_step(
        params0, grads, None, lr=LR, momentum=MOMENTUM, weight_decay=DECAY
    )
    decayed = [g + DECAY * p for p, g in zip(params0, grads, strict=True)]
    _assert_lists_match(
        velocity, decayed, "decay must be added to the gradient before the buffer is formed"
    )
    _assert_lists_match(
        new_params,
        [p - LR * (d + MOMENTUM * d) for p, d in zip(params0, decayed, strict=True)],
        "first Nesterov step must look ahead by momentum times the first buffer",
    )


def test_velocity_state_is_the_unscaled_buffer_including_decay():
    params0 = _tensors(9)
    grad_steps = _grad_steps(9, steps=5)
    _, expected = _torch_reference(params0, grad_steps, 0.3, MOMENTUM, DECAY)
    _, actual = _submission_trace(params0, grad_steps, 0.3, MOMENTUM, DECAY)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(
            got,
            want,
            "velocity state must be the unscaled buffer built from gradient plus decay",
        )


def test_zero_weight_decay_is_plain_nesterov():
    params0 = _tensors(13)
    grad_steps = _grad_steps(13, steps=4)
    expected, _ = _torch_reference(params0, grad_steps, LR, MOMENTUM, 0.0)
    actual, _ = _submission_trace(params0, grad_steps, LR, MOMENTUM, 0.0)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(got, want, "with zero decay the update must match plain Nesterov")


def test_inputs_are_not_mutated():
    params = _tensors(17)
    grads = _grad_steps(17, steps=1)[0]
    velocity = _tensors(18)
    originals = [t.clone() for t in params + grads + velocity]
    sgd_nesterov_step(params, grads, velocity, lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
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
        "zero momentum is not Nesterov",
        "momentum equal to one",
        "negative weight decay",
        "non-finite weight decay",
        "negative learning rate",
        "parameter and gradient counts differ",
    ],
)
def test_invalid_arguments_raise_value_error(case):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    calls = {
        "zero momentum is not Nesterov": lambda: sgd_nesterov_step(
            p, g, None, lr=LR, momentum=0.0, weight_decay=DECAY
        ),
        "momentum equal to one": lambda: sgd_nesterov_step(
            p, g, None, lr=LR, momentum=1.0, weight_decay=DECAY
        ),
        "negative weight decay": lambda: sgd_nesterov_step(
            p, g, None, lr=LR, momentum=MOMENTUM, weight_decay=-0.1
        ),
        "non-finite weight decay": lambda: sgd_nesterov_step(
            p, g, None, lr=LR, momentum=MOMENTUM, weight_decay=float("inf")
        ),
        "negative learning rate": lambda: sgd_nesterov_step(
            p, g, None, lr=-0.1, momentum=MOMENTUM, weight_decay=DECAY
        ),
        "parameter and gradient counts differ": lambda: sgd_nesterov_step(
            p, [], None, lr=LR, momentum=MOMENTUM, weight_decay=DECAY
        ),
    }
    _rejects(calls[case], case)


@pytest.mark.parametrize("scale", [1e-8, 1e8], ids=["tiny-values", "huge-values"])
def test_scale_extremes_match_torch(scale):
    params0 = _tensors(23, scale=scale)
    grad_steps = _grad_steps(23, steps=4, scale=scale)
    expected, _ = _torch_reference(params0, grad_steps, LR, MOMENTUM, DECAY)
    actual, _ = _submission_trace(params0, grad_steps, LR, MOMENTUM, DECAY)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(got, want, "update differs from torch at this magnitude scale")


def test_empty_parameter_list_returns_empty_lists():
    params, velocity = sgd_nesterov_step([], [], None, lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    if list(params) != [] or list(velocity) != []:
        pytest.fail("an empty parameter list must return two empty lists", pytrace=False)


def test_float32_parameters_keep_their_dtype():
    p = [torch.ones(2, 2, dtype=torch.float32)]
    g = [torch.full((2, 2), 0.5, dtype=torch.float32)]
    new_params, velocity = sgd_nesterov_step(
        p, g, None, lr=LR, momentum=MOMENTUM, weight_decay=DECAY
    )
    if new_params[0].dtype != torch.float32 or velocity[0].dtype != torch.float32:
        pytest.fail("output dtype must match the input parameter dtype", pytrace=False)


def _must_raise_value_error(message, call):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{message}: raised {type(exc).__name__} instead of ValueError", pytrace=False)
    pytest.fail(f"{message}: invalid input was accepted", pytrace=False)


def test_zero_learning_rate_is_a_valid_step_that_keeps_parameters():
    params0 = _tensors(33)
    grads = _grad_steps(33, steps=1)[0]
    try:
        new_params, _ = sgd_nesterov_step(
            params0, grads, None, lr=0.0, momentum=MOMENTUM, weight_decay=DECAY
        )
    except Exception as exc:
        pytest.fail(
            f"lr = 0 is allowed (lr >= 0), but a valid zero-rate step raised {type(exc).__name__}",
            pytrace=False,
        )
    _assert_lists_match(
        new_params, params0, "lr = 0 is allowed: the parameters must not move on a zero-rate step"
    )


def test_gradient_with_wrong_shape_raises_value_error():
    params = [torch.zeros(3, dtype=torch.float64)]
    grads = [torch.zeros(1, dtype=torch.float64)]
    _must_raise_value_error(
        "a gradient whose shape differs from its parameter must raise ValueError",
        lambda: sgd_nesterov_step(
            params, grads, None, lr=LR, momentum=MOMENTUM, weight_decay=DECAY
        ),
    )


def test_velocity_buffer_with_wrong_shape_raises_value_error():
    params = [torch.zeros(3, dtype=torch.float64)]
    grads = [torch.zeros(3, dtype=torch.float64)]
    velocity = [torch.zeros(1, dtype=torch.float64)]
    _must_raise_value_error(
        "a velocity buffer whose shape differs from its parameter must raise ValueError",
        lambda: sgd_nesterov_step(
            params, grads, velocity, lr=LR, momentum=MOMENTUM, weight_decay=DECAY
        ),
    )


def test_velocity_with_too_few_buffers_raises_value_error():
    params = [torch.zeros(3, dtype=torch.float64)]
    grads = [torch.zeros(3, dtype=torch.float64)]
    _must_raise_value_error(
        "velocity must hold one buffer per parameter: an empty list must raise ValueError",
        lambda: sgd_nesterov_step(params, grads, [], lr=LR, momentum=MOMENTUM, weight_decay=DECAY),
    )
