# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of
# mistake and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from submission import load_balance_loss as _learner


def _call(logits, alpha=1.0):
    try:
        result = _learner(logits, alpha)
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")
    if not isinstance(result, torch.Tensor):
        pytest.fail("return a 0-dimensional torch.float64 tensor, not a Python float or other type")
    return result


def _invalid(exc_type, logits, alpha=1.0):
    try:
        _learner(logits, alpha)
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument raised {type(exc).__name__}, expected {exc_type.__name__}")
    pytest.fail(f"invalid argument was accepted instead of raising {exc_type.__name__}")


def _reference(rows, alpha):
    """Pure-Python loss. Softmax by hand with the max subtracted; top-1 by explicit tie rule."""
    T, E = len(rows), len(rows[0])
    P = [0.0] * E
    f = [0.0] * E
    for r in rows:
        top = max(r)
        exps = [math.exp(x - top) for x in r]
        total = sum(exps)
        for i in range(E):
            P[i] += (exps[i] / total) / T
        winner = min(range(E), key=lambda i: (-r[i], i))
        f[winner] += 1.0 / T
    return alpha * E * sum(f[i] * P[i] for i in range(E))


def test_seeded_random_logits_match_pure_python_reference():
    gen = torch.Generator().manual_seed(201)
    for tokens, experts, alpha in [(1, 2, 1.0), (4, 3, 1.0), (6, 5, 0.25), (9, 4, 2.0)]:
        logits = torch.randn(tokens, experts, generator=gen, dtype=torch.float64)
        want = _reference(logits.tolist(), alpha)
        got = _call(logits, alpha)
        torch.testing.assert_close(
            got,
            torch.tensor(want, dtype=torch.float64),
            atol=1e-9,
            rtol=1e-9,
            msg="loss disagrees with the reference: check the factor E, the mean over tokens, "
            "and that f uses top-1 counts",
        )


def test_gradient_matches_finite_differences_through_probabilities():
    # Margins between the top two logits in every row are far larger than the perturbation, so
    # the top-1 choice does not change during the finite-difference sweep.
    base = torch.tensor(
        [[0.3, -0.7, 1.1], [1.4, 0.2, -0.5], [-0.2, 0.9, 0.4], [0.1, -1.3, 0.6]],
        dtype=torch.float64,
    )
    alpha = 1.5
    x = base.clone().requires_grad_(True)
    loss = _call(x, alpha)
    if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
        pytest.fail(
            "loss is not connected to the logits: keep P in the autograd graph (do not detach "
            "it, and do not build it from Python floats or .item())"
        )
    (grad,) = torch.autograd.grad(loss, x)
    eps = 1e-6
    fd = torch.zeros_like(base)
    for t in range(base.shape[0]):
        for i in range(base.shape[1]):
            up = base.clone()
            down = base.clone()
            up[t, i] += eps
            down[t, i] -= eps
            fd[t, i] = (_call(up, alpha) - _call(down, alpha)) / (2 * eps)
    torch.testing.assert_close(
        grad,
        fd,
        atol=1e-7,
        rtol=1e-6,
        msg="gradient disagrees with finite differences: the loss must be differentiable "
        "through the router probabilities, and f must stay a constant",
    )


def test_balanced_router_scores_alpha_for_any_alpha():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(
        _call(logits, 0.5),
        torch.tensor(0.5, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="a balanced router must score exactly alpha: check the factor E",
    )


def test_fully_collapsed_router_scores_alpha_times_num_experts():
    logits = torch.tensor([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        _call(logits),
        torch.tensor(3.0, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="a collapsed router must score alpha * E: the factor E is missing or misplaced",
    )


def test_top_one_tie_goes_to_lower_expert_index():
    # Row 0 ties experts 0 and 1. The lower index gives f = [1, 0]; the higher index would give
    # f = [0, 1] and a loss of 1.0 instead of about 1.49995.
    logits = torch.tensor([[0.0, 0.0], [5.0, -5.0]], dtype=torch.float64)
    torch.testing.assert_close(
        _call(logits),
        torch.tensor(1.4999546021312975, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="top-1 ties must go to the lower expert index: check the tie order in f",
    )


def test_single_expert_scores_alpha_for_any_tokens():
    gen = torch.Generator().manual_seed(202)
    logits = torch.randn(3, 1, generator=gen, dtype=torch.float64)
    torch.testing.assert_close(
        _call(logits, 2.0),
        torch.tensor(2.0, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="with one expert, f and P are both 1 and the loss is alpha",
    )


def test_large_logit_magnitudes_stay_finite():
    logits = torch.tensor([[800.0, -800.0], [-800.0, 800.0]], dtype=torch.float64)
    loss = _call(logits)
    assert torch.isfinite(loss).all(), (
        "large logits produced a non-finite loss: softmax must subtract the max"
    )
    torch.testing.assert_close(
        loss,
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="saturated but balanced routing must still score alpha",
    )


def test_loss_scales_linearly_with_alpha_and_zero_alpha_gives_zero():
    gen = torch.Generator().manual_seed(203)
    logits = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    one = _call(logits, 1.0)
    torch.testing.assert_close(
        _call(logits, 2.0),
        2 * one,
        atol=1e-12,
        rtol=1e-12,
        msg="the loss must scale linearly with alpha",
    )
    assert float(_call(logits, 0)) == 0.0, "alpha = 0 must switch the penalty off"


def test_output_is_zero_dimensional_float64():
    gen = torch.Generator().manual_seed(204)
    loss = _call(torch.randn(4, 3, generator=gen, dtype=torch.float64))
    assert isinstance(loss, torch.Tensor), "the loss must be a torch tensor, not a Python float"
    assert loss.dtype == torch.float64, "the loss must be float64 for float64 logits"
    assert tuple(loss.shape) == (), "the loss must be a 0-dimensional tensor"


def test_bool_alpha_raises_type_error():
    _invalid(TypeError, torch.zeros(2, 2, dtype=torch.float64), True)


def test_string_alpha_raises_type_error():
    _invalid(TypeError, torch.zeros(2, 2, dtype=torch.float64), "1.0")


def test_negative_alpha_raises_value_error():
    _invalid(ValueError, torch.zeros(2, 2, dtype=torch.float64), -0.5)


def test_empty_token_axis_raises_value_error():
    _invalid(ValueError, torch.zeros(0, 3, dtype=torch.float64))


def test_one_dimensional_logits_raise_value_error():
    _invalid(ValueError, torch.zeros(4, dtype=torch.float64))


def test_float32_logits_raise_value_error():
    _invalid(ValueError, torch.zeros(2, 3, dtype=torch.float32))


def test_nan_logit_raises_value_error():
    logits = torch.zeros(2, 3, dtype=torch.float64)
    logits[0, 1] = math.nan
    _invalid(ValueError, logits)


def test_inputs_are_not_mutated():
    gen = torch.Generator().manual_seed(205)
    logits = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    before = logits.clone()
    _call(logits, 1.0)
    assert torch.equal(logits, before), "the logits tensor must not be modified in place"


def test_zero_experts_raises_value_error():
    _invalid(ValueError, torch.zeros(3, 0, dtype=torch.float64))


def test_top_one_tie_in_a_wide_router_dispatches_to_lowest_index():
    # Token 0 has equal logits on all 200 experts, so every expert ties for top-1 and the tie
    # goes to expert 0. Token 1 clearly prefers expert 100. So f = 1/2 at expert 0 and 1/2 at
    # expert 100, and 0 elsewhere. The loss is alpha * E * sum_i f_i * P_i with alpha = 1.
    logits = torch.zeros(2, 200, dtype=torch.float64)
    logits[1, 100] = 1.0
    p = torch.softmax(logits, dim=-1).mean(dim=0)  # P_i, the mean router probability
    expected = 200 * (0.5 * p[0] + 0.5 * p[100])
    loss = _call(logits)
    torch.testing.assert_close(
        loss,
        expected,
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "a top-1 tie in a wide router must count toward the lowest expert index: "
            "check the tie order when there are many experts"
        ),
    )
