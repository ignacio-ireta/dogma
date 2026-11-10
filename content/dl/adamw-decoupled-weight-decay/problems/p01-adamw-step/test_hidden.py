from __future__ import annotations

import pytest
import torch
from submission import adamw_step

SHAPES = [(3,), (2, 4), (1, 1, 3)]
LR = 0.05


def _tensors(seed: int, scale: float = 1.0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=gen, dtype=torch.float64) * scale for s in SHAPES]


def _grad_steps(seed: int, steps: int, scale: float = 1.0) -> list[list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed + 1000)
    return [
        [torch.randn(s, generator=gen, dtype=torch.float64) * scale for s in SHAPES]
        for _ in range(steps)
    ]


def _torch_reference(params0, grad_steps, lr, weight_decay, eps=1e-8):
    """torch.optim.AdamW, traced after every step."""
    params = [p.clone().requires_grad_(True) for p in params0]
    optimizer = torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=weight_decay
    )
    trace = []
    for grads in grad_steps:
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        optimizer.step()
        trace.append([p.detach().clone() for p in params])
    return trace


def _submission_trace(params0, grad_steps, lr, weight_decay, eps=1e-8):
    params = [p.clone() for p in params0]
    state = None
    trace = []
    for grads in grad_steps:
        try:
            params, state = adamw_step(
                params,
                [g.clone() for g in grads],
                state,
                lr=lr,
                eps=eps,
                weight_decay=weight_decay,
            )
        except Exception:
            pytest.fail(
                "a valid step raised an error: check the signature and the (m, v, t) state",
                pytrace=False,
            )
        trace.append(params)
    return trace, state


def _assert_lists_match(actual, expected, message: str) -> None:
    for got, want in zip(actual, expected, strict=True):
        if got.shape != want.shape:
            pytest.fail(f"{message}: shape differs from the parameter shape", pytrace=False)
        if got.dtype != want.dtype:
            pytest.fail("output dtype must match the input parameter dtype", pytrace=False)
        if not torch.allclose(got, want, rtol=1e-10, atol=1e-12):
            pytest.fail(message, pytrace=False)


@pytest.mark.parametrize(
    ("seed", "weight_decay", "lr"),
    [(0, 0.1, 0.05), (1, 0.5, 0.1), (2, 0.01, 0.05)],
    ids=["run-a", "run-strong-decay", "run-default-decay"],
)
def test_matches_torch_adamw_over_several_steps(seed, weight_decay, lr):
    params0 = _tensors(seed)
    grad_steps = _grad_steps(seed, steps=20)
    expected = _torch_reference(params0, grad_steps, lr, weight_decay)
    actual, _ = _submission_trace(params0, grad_steps, lr, weight_decay)
    for step, (got, want) in enumerate(zip(actual, expected, strict=True), 1):
        _assert_lists_match(
            got,
            want,
            f"step {step}: update differs from torch.optim.AdamW (check decay order and factor)",
        )


def test_long_run_stays_on_torch_trajectory():
    params0 = _tensors(31)
    grad_steps = _grad_steps(31, steps=200)
    expected = _torch_reference(params0, grad_steps, LR, 0.1)[-1]
    actual, _ = _submission_trace(params0, grad_steps, LR, 0.1)
    _assert_lists_match(actual[-1], expected, "after 200 steps the trajectory left torch's AdamW")


def test_zero_gradient_shrinks_weights_by_lr_times_decay_not_adam_l2():
    # Adam with L2 folds wd * p into the gradient, and the adaptive step would move the weight
    # by about lr toward zero. AdamW must instead multiply by (1 - lr * wd) exactly.
    p0 = [torch.full((2,), 0.01, dtype=torch.float64), torch.full((2,), 2.0, dtype=torch.float64)]
    zeros = [torch.zeros_like(p) for p in p0]
    new_params, _ = adamw_step(p0, zeros, None, lr=0.1, weight_decay=0.5)
    _assert_lists_match(
        new_params,
        [p * (1 - 0.1 * 0.5) for p in p0],
        "zero-gradient weights must shrink by (1 - lr * wd): decay looks folded into the gradient",
    )


def test_weight_decay_is_scaled_by_the_learning_rate():
    p0 = [torch.full((3,), 4.0, dtype=torch.float64)]
    zeros = [torch.zeros(3, dtype=torch.float64)]
    new_params, _ = adamw_step(p0, zeros, None, lr=0.2, weight_decay=0.25)
    _assert_lists_match(
        new_params,
        [p0[0] * (1 - 0.2 * 0.25)],
        "decay factor must be 1 - lr * weight_decay; the learning rate was left out",
    )


def test_zero_weight_decay_matches_plain_adam_oracle():
    params0 = _tensors(11)
    grad_steps = _grad_steps(11, steps=6)
    expected = _torch_reference(params0, grad_steps, LR, 0.0)
    actual, _ = _submission_trace(params0, grad_steps, LR, 0.0)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(got, want, "with zero decay AdamW must reduce to Adam")


def test_first_step_counter_is_one_and_inputs_are_not_mutated():
    params = _tensors(17)
    grads = _grad_steps(17, steps=1)[0]
    originals = [t.clone() for t in params + grads]
    _, state = adamw_step(params, grads, None, lr=LR, weight_decay=0.1)
    if isinstance(state[2], bool) or state[2] != 1:
        pytest.fail(
            "step counter must start at zero and read 1 after the first update", pytrace=False
        )
    for before, after in zip(originals, params + grads, strict=True):
        if not torch.equal(before, after):
            pytest.fail("input tensors were mutated: return new tensors instead", pytrace=False)


@pytest.mark.parametrize("scale", [1e-6, 1e4], ids=["tiny-weights", "large-weights"])
def test_scale_extremes_match_torch(scale):
    params0 = _tensors(23, scale=scale)
    grad_steps = _grad_steps(23, steps=5, scale=scale)
    expected = _torch_reference(params0, grad_steps, LR, 0.1)
    actual, _ = _submission_trace(params0, grad_steps, LR, 0.1)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(got, want, "update differs from torch at this magnitude scale")


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
        "negative weight decay",
        "non-finite weight decay",
        "negative learning rate",
        "beta2 equal to one",
        "gradient shape differs from its parameter",
    ],
    ids=[
        "negative-weight-decay",
        "non-finite-weight-decay",
        "negative-learning-rate",
        "beta-two-equal-to-one",
        "gradient-shape-mismatch",
    ],
)
def test_invalid_arguments_raise_value_error(case):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    calls = {
        "negative weight decay": lambda: adamw_step(p, g, None, lr=LR, weight_decay=-0.1),
        "non-finite weight decay": lambda: adamw_step(p, g, None, lr=LR, weight_decay=float("nan")),
        "negative learning rate": lambda: adamw_step(p, g, None, lr=-LR, weight_decay=0.1),
        "beta2 equal to one": lambda: adamw_step(p, g, None, lr=LR, beta2=1.0, weight_decay=0.1),
        "gradient shape differs from its parameter": lambda: adamw_step(
            p, [torch.zeros(2, dtype=torch.float64)], None, lr=LR, weight_decay=0.1
        ),
    }
    _rejects(calls[case], case)


def test_float32_parameters_keep_their_dtype():
    p = [torch.ones(2, 2, dtype=torch.float32)]
    g = [torch.full((2, 2), 0.5, dtype=torch.float32)]
    new_params, state = adamw_step(p, g, None, lr=LR, weight_decay=0.1)
    if new_params[0].dtype != torch.float32 or state[0][0].dtype != torch.float32:
        pytest.fail("output dtype must match the input parameter dtype", pytrace=False)


def _state_rejects(m, v, t, case: str) -> None:
    zeros = torch.zeros(3, dtype=torch.float64)
    _rejects(
        lambda: adamw_step([zeros], [zeros], (m, v, t), lr=LR, weight_decay=0.1),
        case,
    )


def test_negative_step_count_in_state_is_rejected():
    zeros = torch.zeros(3, dtype=torch.float64)
    _state_rejects([zeros], [zeros], -1, "negative step count in the carried-over state")


def test_boolean_step_count_in_state_is_rejected():
    zeros = torch.zeros(3, dtype=torch.float64)
    _state_rejects([zeros], [zeros], True, "boolean step count in the carried-over state")


def test_first_moment_of_wrong_shape_in_state_is_rejected():
    zeros = torch.zeros(3, dtype=torch.float64)
    bad = torch.zeros(1, dtype=torch.float64)
    _state_rejects([bad], [zeros], 0, "first moment in the carried-over state has the wrong shape")


def test_second_moment_of_wrong_shape_in_state_is_rejected():
    zeros = torch.zeros(3, dtype=torch.float64)
    bad = torch.zeros(1, dtype=torch.float64)
    _state_rejects([zeros], [bad], 0, "second moment in the carried-over state has the wrong shape")


@pytest.mark.parametrize("which", ["first", "second"], ids=["first-moment", "second-moment"])
def test_moment_wrong_shape_on_one_parameter_of_several_is_rejected(which):
    zeros = torch.zeros(3, dtype=torch.float64)
    bad = torch.zeros(1, dtype=torch.float64)
    p = [zeros, zeros]
    g = [zeros, zeros]
    good_pair = [zeros, zeros]
    bad_pair = [zeros, bad]
    m, v = (bad_pair, good_pair) if which == "first" else (good_pair, bad_pair)
    _rejects(
        lambda: adamw_step(p, g, (m, v, 0), lr=LR, weight_decay=0.1),
        f"{which} moment has the wrong shape for only one of several parameters",
    )


@pytest.mark.parametrize(
    "eps", [0.0, -1e-8, float("nan"), float("inf")], ids=["zero", "negative", "nan", "inf"]
)
def test_eps_must_be_finite_and_positive(eps):
    zeros = torch.zeros(3, dtype=torch.float64)
    _rejects(
        lambda: adamw_step([zeros], [zeros], None, lr=LR, eps=eps, weight_decay=0.1),
        "eps outside its allowed range (must be finite and positive)",
    )


def test_carried_over_first_moment_is_not_changed_by_the_update():
    # the moments in a carried-over state are inputs and must keep their values
    params = [torch.tensor([1.0, -2.0], dtype=torch.float64)]
    _, state = adamw_step(params, [torch.tensor([0.5, 3.0], dtype=torch.float64)], None, lr=0.1)
    m_before = [t.clone() for t in state[0]]
    adamw_step(params, [torch.tensor([-1.0, 2.0], dtype=torch.float64)], state, lr=0.1)
    torch.testing.assert_close(
        state[0][0],
        m_before[0],
        atol=0,
        rtol=0,
        msg="the carried-over first moment must not be changed in place",
    )


def test_moments_keep_the_bfloat16_dtype_of_their_parameter():
    # the returned moments follow the input dtype, not the float32 default
    p = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    g = torch.tensor([0.5, 3.0], dtype=torch.bfloat16)
    _, (new_m, new_v, _) = adamw_step([p], [g], None, lr=0.1)
    assert new_m[0].dtype == torch.bfloat16, "the first moment must have the dtype of its parameter"
    assert new_v[0].dtype == torch.bfloat16, (
        "the second moment must have the dtype of its parameter"
    )


def test_zero_learning_rate_is_accepted_and_leaves_weights_unchanged():
    # lr = 0 is a valid non-negative rate: shrink = 1 and the Adam step is multiplied by 0
    params = [torch.tensor([1.0, -2.0], dtype=torch.float64)]
    grads = [torch.tensor([0.5, 3.0], dtype=torch.float64)]
    try:
        new_params, _ = adamw_step(params, grads, None, lr=0.0, weight_decay=0.5)
    except ValueError:
        raise AssertionError("a zero learning rate is a valid input and must be accepted") from None
    torch.testing.assert_close(
        new_params[0],
        params[0],
        atol=0,
        rtol=0,
        msg="a zero learning rate must be accepted and must leave the weights unchanged",
    )


def test_zero_dimensional_parameter_is_updated_like_torch_adamw():
    # a scalar weight is a floating-point tensor of shape () and must be accepted
    p = torch.tensor(1.0, dtype=torch.float64)
    g = torch.tensor(0.5, dtype=torch.float64)
    ref = torch.nn.Parameter(p.clone())
    optimizer = torch.optim.AdamW([ref], lr=0.1, weight_decay=0.5)
    ref.grad = g.clone()
    optimizer.step()
    try:
        new_params, _ = adamw_step([p], [g], None, lr=0.1, weight_decay=0.5)
    except ValueError:
        raise AssertionError(
            "a zero-dimensional parameter is a valid tensor and must be accepted"
        ) from None
    torch.testing.assert_close(
        new_params[0],
        ref.detach(),
        atol=1e-10,
        rtol=1e-10,
        msg="a zero-dimensional parameter must be updated as torch.optim.AdamW updates it",
    )
