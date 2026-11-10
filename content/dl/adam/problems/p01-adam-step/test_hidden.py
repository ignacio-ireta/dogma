from __future__ import annotations

import pytest
import torch
from submission import adam_step

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


def _torch_reference(params0, grad_steps, lr, eps):
    """torch.optim.Adam (default betas, no weight decay), traced after every step."""
    params = [p.clone().requires_grad_(True) for p in params0]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=eps)
    trace = []
    for grads in grad_steps:
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        optimizer.step()
        trace.append([p.detach().clone() for p in params])
    return trace


def _submission_trace(params0, grad_steps, lr, eps):
    params = [p.clone() for p in params0]
    state = None
    trace = []
    for grads in grad_steps:
        try:
            params, state = adam_step(
                params,
                [g.clone() for g in grads],
                state,
                lr=lr,
                eps=eps,
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
    ("seed", "eps"),
    [(0, 1e-8), (1, 1e-8), (2, 1e-2)],
    ids=["run-a", "run-b", "run-large-eps"],
)
def test_matches_torch_adam_over_several_steps(seed, eps):
    params0 = _tensors(seed)
    grad_steps = _grad_steps(seed, steps=20)
    expected = _torch_reference(params0, grad_steps, LR, eps)
    actual, _ = _submission_trace(params0, grad_steps, LR, eps)
    for step, (got, want) in enumerate(zip(actual, expected, strict=True), 1):
        _assert_lists_match(
            got,
            want,
            f"step {step}: update differs from torch.optim.Adam (check bias correction and eps)",
        )


def test_long_run_stays_on_torch_trajectory():
    params0 = _tensors(31)
    grad_steps = _grad_steps(31, steps=200)
    expected = _torch_reference(params0, grad_steps, LR, 1e-8)[-1]
    actual, _ = _submission_trace(params0, grad_steps, LR, 1e-8)
    _assert_lists_match(actual[-1], expected, "after 200 steps the trajectory left torch's Adam")


def test_first_step_matches_bias_corrected_formula():
    gen = torch.Generator().manual_seed(41)
    p0 = [torch.randn(3, generator=gen, dtype=torch.float64)]
    g = [torch.randn(3, generator=gen, dtype=torch.float64)]
    new_params, _ = adam_step(p0, g, None, lr=LR)
    if not all(torch.isfinite(x).all() for x in new_params):
        pytest.fail(
            "first update produced non-finite values: the step counter must be incremented "
            "before bias correction, so the first update divides by 1 - beta**1",
            pytrace=False,
        )
    # With t = 1: m_hat = g and v_hat = g * g, so the step is lr * g / (|g| + eps).
    expected = [p - LR * x / (x.abs() + 1e-8) for p, x in zip(p0, g, strict=True)]
    _assert_lists_match(
        new_params,
        expected,
        "first step differs from lr * g / (|g| + eps): bias correction is missing or wrong",
    )


def test_step_counter_starts_at_zero_and_counts_each_update():
    params = _tensors(3)
    state = None
    for expected_count in (1, 2, 3):
        params, state = adam_step(params, _grad_steps(3, steps=1)[0], state, lr=LR)
        count = state[2]
        if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
            pytest.fail(
                "step counter must start at zero and count updates: the first update uses t = 1",
                pytrace=False,
            )


def test_eps_is_added_after_the_square_root_and_bias_correction():
    p0 = [torch.zeros(2, dtype=torch.float64)]
    g = [torch.full((2,), 1e-3, dtype=torch.float64)]
    new_params, _ = adam_step(p0, g, None, lr=0.1, eps=1e-2)
    # m_hat = g, v_hat = g * g, so the denominator is |g| + eps.
    expected = [-0.1 * 1e-3 / (1e-3 + 1e-2) * torch.ones(2, dtype=torch.float64)]
    _assert_lists_match(
        new_params,
        expected,
        "eps must be added after the square root (and after bias correction), not inside it",
    )


def test_zero_gradient_leaves_parameters_unchanged_and_finite():
    params = _tensors(7)
    zeros = [torch.zeros_like(p) for p in params]
    new_params, _ = adam_step(params, zeros, None, lr=LR)
    for got in new_params:
        if not torch.isfinite(got).all():
            pytest.fail(
                "zero gradient produced non-finite values: check the counter and eps", pytrace=False
            )
    _assert_lists_match(new_params, params, "zero gradient must leave parameters unchanged")


def test_inputs_are_not_mutated():
    params = _tensors(17)
    grads = _grad_steps(17, steps=1)[0]
    _, state = adam_step(params, grads, None, lr=LR)
    originals = [t.clone() for t in params + grads + (state[0] + state[1])]
    adam_step(params, grads, state, lr=LR)
    for before, after in zip(originals, params + grads + (state[0] + state[1]), strict=True):
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
        "beta1 equal to one",
        "negative beta2",
        "zero eps",
        "negative learning rate",
        "moment list length differs from parameter count",
        "non-integer step count",
        "boolean step count",
        "gradient shape differs from its parameter",
    ],
    ids=[
        "beta-equal-to-one",
        "negative-beta",
        "zero-eps",
        "negative-learning-rate",
        "moment-length-mismatch",
        "non-integer-step",
        "boolean-step",
        "gradient-shape-mismatch",
    ],
)
def test_invalid_arguments_and_state_raise_value_error(case):
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.zeros(3, dtype=torch.float64)]
    good_state = ([torch.zeros(3, dtype=torch.float64)], [torch.zeros(3, dtype=torch.float64)], 1)
    calls = {
        "beta1 equal to one": lambda: adam_step(p, g, None, lr=LR, beta1=1.0),
        "negative beta2": lambda: adam_step(p, g, None, lr=LR, beta2=-0.1),
        "zero eps": lambda: adam_step(p, g, None, lr=LR, eps=0.0),
        "negative learning rate": lambda: adam_step(p, g, None, lr=-LR),
        "moment list length differs from parameter count": lambda: adam_step(
            p, g, ([], [], 1), lr=LR
        ),
        "non-integer step count": lambda: adam_step(
            p, g, (good_state[0], good_state[1], 1.0), lr=LR
        ),
        "boolean step count": lambda: adam_step(p, g, (good_state[0], good_state[1], True), lr=LR),
        "gradient shape differs from its parameter": lambda: adam_step(
            p, [torch.zeros(2, dtype=torch.float64)], None, lr=LR
        ),
    }
    _rejects(calls[case], case)


@pytest.mark.parametrize("scale", [1e-6, 1e6], ids=["tiny-gradients", "huge-gradients"])
def test_scale_extremes_match_torch(scale):
    params0 = _tensors(23, scale=scale)
    grad_steps = _grad_steps(23, steps=5, scale=scale)
    expected = _torch_reference(params0, grad_steps, LR, 1e-8)
    actual, _ = _submission_trace(params0, grad_steps, LR, 1e-8)
    for got, want in zip(actual, expected, strict=True):
        _assert_lists_match(got, want, "update differs from torch at this magnitude scale")


def test_float32_parameters_keep_their_dtype():
    p = [torch.ones(2, 2, dtype=torch.float32)]
    g = [torch.full((2, 2), 0.5, dtype=torch.float32)]
    new_params, state = adam_step(p, g, None, lr=LR)
    if new_params[0].dtype != torch.float32 or state[0][0].dtype != torch.float32:
        pytest.fail("output dtype must match the input parameter dtype", pytrace=False)


def test_zero_learning_rate_is_valid_and_leaves_parameters_unchanged():
    p = [torch.tensor([1.0, -2.0], dtype=torch.float64)]
    g = [torch.tensor([0.5, 3.0], dtype=torch.float64)]
    new_params, _ = adam_step(p, g, None, lr=0.0)
    _assert_lists_match(
        new_params,
        p,
        "a learning rate of zero is valid and must leave the parameters unchanged",
    )


def test_zero_beta2_is_valid_and_gives_the_squared_gradient_as_second_moment():
    p = [torch.zeros(3, dtype=torch.float64)]
    g = [torch.tensor([2.0, -0.5, 1.0], dtype=torch.float64)]
    new_params, _ = adam_step(p, g, None, lr=0.1, beta2=0.0)
    expected = [p[0] - 0.1 * g[0] / (g[0].abs() + 1e-8)]
    _assert_lists_match(
        new_params,
        expected,
        "beta2 of zero is valid: with t = 1 the second moment is the squared gradient",
    )


def test_first_moment_with_a_wrong_shape_is_rejected():
    params = [torch.zeros(3, dtype=torch.float64), torch.zeros(2, 4, dtype=torch.float64)]
    grads = [torch.ones_like(p) for p in params]
    state = (
        [torch.zeros(3, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64)],
        [torch.zeros_like(p) for p in params],
        1,
    )
    _rejects(
        lambda: adam_step(params, grads, state, lr=LR),
        "a first moment whose shape differs from its parameter must raise ValueError",
    )


def test_second_moment_with_a_wrong_shape_is_rejected():
    params = [torch.zeros(3, dtype=torch.float64), torch.zeros(2, 4, dtype=torch.float64)]
    grads = [torch.ones_like(p) for p in params]
    state = (
        [torch.zeros_like(p) for p in params],
        [torch.zeros(3, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64)],
        1,
    )
    _rejects(
        lambda: adam_step(params, grads, state, lr=LR),
        "a second moment whose shape differs from its parameter must raise ValueError",
    )


def test_zero_beta1_is_valid_and_gives_the_gradient_as_first_moment():
    # beta1 = 0 keeps no history, so m = g and m_hat = g / (1 - 0**1) = g.
    # With g = 2 and t = 1: v = 0.001 * 4, v_hat = 4, sqrt(v_hat) = 2,
    # step = lr * 2 / 2 = lr, so the parameter moves from 1.0 to 0.9.
    try:
        params, state = adam_step(
            [torch.tensor([1.0], dtype=torch.float64)],
            [torch.tensor([2.0], dtype=torch.float64)],
            None,
            lr=0.1,
            beta1=0.0,
        )
    except ValueError:
        pytest.fail("beta1 = 0 must be accepted: it is inside the range [0, 1)", pytrace=False)
    torch.testing.assert_close(
        params[0],
        torch.tensor([0.9], dtype=torch.float64),
        atol=1e-7,
        rtol=0,
        msg="beta1 = 0 must be accepted: the first moment is then the gradient itself",
    )
    torch.testing.assert_close(
        state[0][0],
        torch.tensor([2.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
        msg="beta1 = 0 must be accepted: the stored first moment must equal the gradient",
    )
