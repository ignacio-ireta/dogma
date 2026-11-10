from __future__ import annotations

import pytest
import torch
from submission import amsgrad_step

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


def _torch_reference(params0, grad_steps, lr, beta2, eps=1e-8):
    """torch.optim.Adam(amsgrad=True), traced after every step."""
    params = [p.clone().requires_grad_(True) for p in params0]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, beta2), eps=eps, amsgrad=True)
    trace = []
    for grads in grad_steps:
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        optimizer.step()
        trace.append([p.detach().clone() for p in params])
    return trace


def _submission_trace(params0, grad_steps, lr, beta2, eps=1e-8):
    params = [p.clone() for p in params0]
    state = None
    trace = []
    for grads in grad_steps:
        try:
            params, state = amsgrad_step(
                params, [g.clone() for g in grads], state, lr=lr, beta2=beta2, eps=eps
            )
        except Exception:
            pytest.fail(
                "a valid step raised an error: check the signature and the (m, v, v_max, t) state",
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
    ("seed", "beta2"),
    [(0, 0.999), (1, 0.999), (2, 0.5)],
    ids=["run-a", "run-b", "run-fast-decay"],
)
def test_matches_torch_amsgrad_over_several_steps(seed, beta2):
    params0 = _tensors(seed)
    grad_steps = _grad_steps(seed, steps=20)
    expected = _torch_reference(params0, grad_steps, LR, beta2)
    actual, _ = _submission_trace(params0, grad_steps, LR, beta2)
    for step, (got, want) in enumerate(zip(actual, expected, strict=True), 1):
        _assert_lists_match(
            got,
            want,
            f"step {step}: update differs from torch.optim.Adam(amsgrad=True)",
        )


def test_denominator_keeps_the_running_maximum_after_gradients_shrink():
    # Large gradients first, then tiny ones: plain Adam forgets the large second moment
    # quickly with beta2 = 0.5, but AMSGrad must keep it in the denominator.
    params0 = [torch.ones(2, dtype=torch.float64)]
    big = [[torch.full((2,), 2.0, dtype=torch.float64)] for _ in range(3)]
    small = [[torch.full((2,), 0.01, dtype=torch.float64)] for _ in range(8)]
    grad_steps = big + small
    expected = _torch_reference(params0, grad_steps, 0.1, beta2=0.5)
    actual, _ = _submission_trace(params0, grad_steps, 0.1, beta2=0.5)
    _assert_lists_match(
        actual[-1],
        expected[-1],
        "denominator must use the running maximum of v: a shrinking gradient "
        "must not shrink the second-moment denominator",
    )


def test_first_update_matches_adam_bias_corrected_step():
    p0 = [torch.tensor([1.0, -2.0], dtype=torch.float64)]
    g = [torch.tensor([0.3, -0.7], dtype=torch.float64)]
    new_params, _ = amsgrad_step(p0, g, None, lr=LR)
    expected = [p0[0] - LR * g[0] / (g[0].abs() + 1e-8)]
    _assert_lists_match(
        new_params,
        expected,
        "first AMSGrad update must equal the first Adam update: bias correction is off",
    )


def test_inputs_are_not_mutated():
    params = _tensors(17)
    grads = _grad_steps(17, steps=1)[0]
    _, state = amsgrad_step(params, grads, None, lr=LR)
    originals = [t.clone() for t in params + grads + (state[0] + state[1] + state[2])]
    amsgrad_step(params, grads, state, lr=LR)
    for before, after in zip(
        originals, params + grads + (state[0] + state[1] + state[2]), strict=True
    ):
        if not torch.equal(before, after):
            pytest.fail("input tensors were mutated: return new tensors instead", pytrace=False)


def test_mismatched_state_lengths_raise_value_error():
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    try:
        amsgrad_step(p, g, ([], [], [], 1), lr=LR)
    except ValueError:
        return
    except Exception:
        pytest.fail("malformed state must raise ValueError, not another exception", pytrace=False)
    pytest.fail(
        "malformed state was accepted: moment lists must match the parameter count", pytrace=False
    )


def test_zero_gradients_leave_parameters_unchanged():
    params = _tensors(7)
    zeros = [torch.zeros_like(p) for p in params]
    new_params, _ = amsgrad_step(params, zeros, None, lr=LR)
    _assert_lists_match(new_params, params, "zero gradient must leave parameters unchanged")


@pytest.mark.parametrize("scale", [1e-6, 1e6], ids=["tiny-gradients", "huge-gradients"])
def test_scale_extremes_match_torch(scale):
    params0 = _tensors(23, scale=scale)
    grad_steps = _grad_steps(23, steps=5, scale=scale)
    expected = _torch_reference(params0, grad_steps, LR, 0.999)
    actual, _ = _submission_trace(params0, grad_steps, LR, 0.999)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(got, want, "update differs from torch at this magnitude scale")


def test_float32_parameters_keep_their_dtype():
    p = [torch.ones(2, 2, dtype=torch.float32)]
    g = [torch.full((2, 2), 0.5, dtype=torch.float32)]
    new_params, state = amsgrad_step(p, g, None, lr=LR)
    if new_params[0].dtype != torch.float32 or state[2][0].dtype != torch.float32:
        pytest.fail("output dtype must match the input parameter dtype", pytrace=False)


def test_maximum_is_taken_over_the_uncorrected_second_moment():
    # torch order: v_max = max(v_max, v) on the raw moment, then only v_max is bias-corrected.
    # Correcting each v before the maximum gives a different denominator on this sequence.
    params0 = [torch.ones(2, dtype=torch.float64)]
    grad_steps = [
        [torch.full((2,), 2.0, dtype=torch.float64)],
        [torch.full((2,), 0.1, dtype=torch.float64)],
    ]
    expected = _torch_reference(params0, grad_steps, 0.1, beta2=0.5)
    actual, _ = _submission_trace(params0, grad_steps, 0.1, beta2=0.5)
    _assert_lists_match(
        actual[-1],
        expected[-1],
        "v_max must be the maximum of the uncorrected second moment, bias-corrected afterwards",
    )


@pytest.mark.parametrize(
    "case",
    [
        "negative learning rate",
        "beta1 equal to one",
        "negative beta2",
        "zero eps",
    ],
    ids=["negative-learning-rate", "beta-equal-to-one", "negative-beta", "zero-eps"],
)
def test_invalid_scalars_raise_value_error(case):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    calls = {
        "negative learning rate": lambda: amsgrad_step(p, g, None, lr=-LR),
        "beta1 equal to one": lambda: amsgrad_step(p, g, None, lr=LR, beta1=1.0),
        "negative beta2": lambda: amsgrad_step(p, g, None, lr=LR, beta2=-0.1),
        "zero eps": lambda: amsgrad_step(p, g, None, lr=LR, eps=0.0),
    }
    try:
        calls[case]()
    except ValueError:
        return
    except Exception:
        pytest.fail(f"{case}: must raise ValueError, not another exception type", pytrace=False)
    pytest.fail(f"{case}: invalid scalar was accepted", pytrace=False)


def _rejects_value_error(call, case):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(f"{case}: must raise ValueError, not another exception type", pytrace=False)
    pytest.fail(f"{case}: invalid input was accepted", pytrace=False)


@pytest.mark.parametrize("grad_size", [1, 2], ids=["one-entry-gradient", "two-entry-gradient"])
def test_gradient_with_a_different_shape_is_rejected(grad_size):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.ones(grad_size, dtype=torch.float64)]
    _rejects_value_error(
        lambda: amsgrad_step(p, g, None, lr=LR),
        "a gradient whose shape differs from its parameter must raise ValueError",
    )


@pytest.mark.parametrize(
    "wrong", ["first", "second", "max"], ids=["first-moment", "second-moment", "max-moment"]
)
def test_state_tensor_with_a_wrong_shape_is_rejected(wrong):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.ones(3, dtype=torch.float64)]
    moments = {name: [torch.zeros(3, dtype=torch.float64)] for name in ("first", "second", "max")}
    moments[wrong] = [torch.zeros(1, dtype=torch.float64)]
    state = (moments["first"], moments["second"], moments["max"], 1)
    _rejects_value_error(
        lambda: amsgrad_step(p, g, state, lr=LR),
        "a state tensor whose shape differs from its parameter must raise ValueError",
    )


@pytest.mark.parametrize("bad_t", [1.0, True, -1], ids=["float", "boolean", "negative"])
def test_malformed_step_count_is_rejected(bad_t):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.ones(3, dtype=torch.float64)]
    state = (
        [torch.zeros(3, dtype=torch.float64)],
        [torch.zeros(3, dtype=torch.float64)],
        [torch.zeros(3, dtype=torch.float64)],
        bad_t,
    )
    _rejects_value_error(
        lambda: amsgrad_step(p, g, state, lr=LR),
        "a step count that is not a non-negative integer must raise ValueError",
    )


def test_zero_step_count_with_zero_moments_is_a_valid_start():
    p = [torch.tensor([1.0, -2.0], dtype=torch.float64)]
    g = [torch.tensor([0.3, -0.7], dtype=torch.float64)]
    zeros = [torch.zeros(2, dtype=torch.float64)]
    fresh, _ = amsgrad_step(p, g, None, lr=LR)
    from_zero_state, state = amsgrad_step(p, g, (zeros, zeros, zeros, 0), lr=LR)
    assert state[3] == 1, "a step count of zero is valid and the update must count one step"
    _assert_lists_match(
        from_zero_state,
        fresh,
        "a state with step count zero and zero moments must behave like no state",
    )


def test_zero_beta1_is_valid_and_gives_the_signed_gradient_step():
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.tensor([2.0, -0.5, 1.0], dtype=torch.float64)]
    new_params, _ = amsgrad_step(p, g, None, lr=0.1, beta1=0.0)
    expected = [p[0] - 0.1 * g[0] / (g[0].abs() + 1e-8)]
    _assert_lists_match(
        new_params,
        expected,
        "beta1 of zero is valid: with t = 1 the first moment is the gradient itself",
    )


def test_beta2_of_one_is_rejected():
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    _rejects_value_error(
        lambda: amsgrad_step(p, g, None, lr=LR, beta2=1.0),
        "beta2 must lie in [0, 1): a value of one must raise ValueError",
    )


def test_negative_beta1_is_rejected():
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    _rejects_value_error(
        lambda: amsgrad_step(p, g, None, lr=LR, beta1=-0.5),
        "beta1 must lie in [0, 1): a value below the range must raise ValueError",
    )
