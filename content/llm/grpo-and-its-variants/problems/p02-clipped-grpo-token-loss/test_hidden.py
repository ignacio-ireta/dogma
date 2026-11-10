# Hidden tests: an independent per-response loop reference, clip and mask behaviour,
# the per-response averaging convention, the k3 KL term, gradients (analytic and
# finite-difference), dtype and the validation rules. The oracle is plain Python floats,
# used only here. Messages name the kind of mistake and never show hidden values.
from __future__ import annotations

import math

import pytest
import torch
from submission import grpo_token_loss

ATOL = 1e-9
RTOL = 1e-9
# float32 keeps about 7 significant digits. Log-ratios of order one and a few tokens per response
# keep the rounding near 1e-6, so 1e-5 is safe and still rejects a wrong clip or a wrong mask.
FLOAT32_ATOL = 1e-5


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(shape, seed, scale=1.0, dtype=torch.float64):
    return (torch.randn(shape, generator=_gen(seed), dtype=dtype) * scale).clone()


def _batch(seed, groups=4, length=6, spread=0.5, dtype=torch.float64):
    """Random ragged batch: each row has at least one response token; log-ratios spread enough "
    "to clip."""
    new = _randn((groups, length), seed, scale=0.3, dtype=dtype) - 1.0
    old = new - _randn((groups, length), seed + 1, scale=spread, dtype=dtype)
    ref = new + _randn((groups, length), seed + 2, scale=0.4, dtype=dtype)
    adv = _randn((groups,), seed + 3, scale=1.0, dtype=dtype)
    gen = _gen(seed + 4)
    mask = torch.rand((groups, length), generator=gen) < 0.7
    mask[:, 0] = True  # every row keeps at least one token
    return new, old, ref, adv, mask


def _reference(new, old, ref, adv, mask, clip, beta):  # noqa: PLR0917
    """Independent oracle: pure-Python loops over responses and tokens, averaged per response."""
    nv, ov, rv = new.tolist(), old.tolist(), ref.tolist()
    av, mv = adv.tolist(), mask.tolist()
    total = 0.0
    for i in range(len(nv)):
        tokens = [t for t, keep in enumerate(mv[i]) if keep]
        acc = 0.0
        for t in tokens:
            rho = math.exp(nv[i][t] - ov[i][t])
            a = av[i]
            clipped = min(max(rho, 1.0 - clip), 1.0 + clip)
            surrogate = min(rho * a, clipped * a)
            d = rv[i][t] - nv[i][t]
            k3 = math.exp(d) - d - 1.0
            acc += -(surrogate - beta * k3)
        total += acc / len(tokens)
    return total / len(nv)


def _call(new, old, ref, adv, mask, **kw):
    snap = [t.clone() for t in (new, old, ref, adv)] + [mask.clone()]
    try:
        out = grpo_token_loss(new, old, ref, adv, mask, **kw)
    except ValueError as exc:
        pytest.fail(
            f"valid input raised ValueError ({exc}): check the contract rules", pytrace=False
        )
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, masks and dtypes",
            pytrace=False,
        )
    if not torch.is_tensor(out) or out.ndim != 0:
        pytest.fail("return a 0-dimensional tensor: the scalar loss", pytrace=False)
    if out.dtype != new.dtype:
        pytest.fail("the loss must keep the dtype of the inputs; do not downcast", pytrace=False)
    current = [new, old, ref, adv, mask]
    for before, after in zip(snap, current, strict=True):
        if not torch.equal(before, after):
            pytest.fail("an input tensor was modified in place; build new tensors", pytrace=False)
    return out


def _close(actual, expected, message, atol=ATOL, rtol=RTOL):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        pytest.fail(message, pytrace=False)


def _expect_value_error(thunk, message):
    try:
        thunk()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail(message, pytrace=False)


PER_RESPONSE = (
    "loss disagrees with the per-response reference: average the masked tokens within each "
    "response "
    "first, then average over the group; check the clip, the advantage sign and the KL term"
)


def test_loss_matches_per_response_reference_on_random_ragged_batches():
    for seed in range(4):
        new, old, ref, adv, mask = _batch(seed * 10)
        out = _call(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04)
        _close(
            out,
            torch.tensor(_reference(new, old, ref, adv, mask, 0.2, 0.04), dtype=torch.float64),
            PER_RESPONSE,
        )


def test_positive_advantage_above_upper_clip_gets_no_policy_gradient():
    new = torch.tensor([[math.log(1.5)]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros((1, 1), dtype=torch.float64)
    ref = new.detach().clone()
    loss = grpo_token_loss(
        new,
        old,
        ref,
        torch.tensor([1.0], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        clip_eps=0.2,
        kl_coef=0.0,
    )
    loss.backward()
    _close(
        new.grad,
        torch.zeros_like(new),
        "a token past 1 + eps with positive advantage still got gradient: the min must keep the "
        "clipped branch",
    )


def test_negative_advantage_below_lower_clip_gets_no_policy_gradient():
    new = torch.tensor([[math.log(0.5)]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros((1, 1), dtype=torch.float64)
    ref = new.detach().clone()
    loss = grpo_token_loss(
        new,
        old,
        ref,
        torch.tensor([-1.0], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        clip_eps=0.2,
        kl_coef=0.0,
    )
    loss.backward()
    _close(
        new.grad,
        torch.zeros_like(new),
        "a token below 1 - eps with negative advantage still got gradient: the min must keep the "
        "clipped branch",
    )


def test_positive_advantage_below_lower_clip_keeps_its_gradient():
    # With A > 0 and ratio below 1 - eps, the unclipped branch is the smaller one, so the min
    # selects it and the token keeps its gradient.
    new = torch.tensor([[math.log(0.5)]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros((1, 1), dtype=torch.float64)
    ref = new.detach().clone()
    loss = grpo_token_loss(
        new,
        old,
        ref,
        torch.tensor([1.0], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        clip_eps=0.2,
        kl_coef=0.0,
    )
    loss.backward()
    # d(-ratio * A)/d(logp_new) = -ratio * A = -0.5 for this token.
    _close(
        new.grad,
        torch.tensor([[-0.5]], dtype=torch.float64),
        "a positive-advantage token below 1 - eps lost its gradient: the min must keep the "
        "unclipped branch when it is smaller",
    )


def test_negative_advantage_above_upper_clip_keeps_its_gradient():
    # With A < 0 and ratio above 1 + eps, the unclipped branch is the smaller one, so it wins.
    new = torch.tensor([[math.log(1.5)]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros((1, 1), dtype=torch.float64)
    ref = new.detach().clone()
    loss = grpo_token_loss(
        new,
        old,
        ref,
        torch.tensor([-1.0], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        clip_eps=0.2,
        kl_coef=0.0,
    )
    loss.backward()
    # d(-ratio * A)/d(logp_new) = -ratio * A = 1.5 for this token.
    _close(
        new.grad,
        torch.tensor([[1.5]], dtype=torch.float64),
        "a negative-advantage token above 1 + eps lost its gradient: the min must keep the "
        "unclipped branch when it is smaller",
    )


def test_unclipped_token_inside_the_band_keeps_its_gradient():
    new = torch.tensor([[math.log(1.05)]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros((1, 1), dtype=torch.float64)
    ref = new.detach().clone()
    loss = grpo_token_loss(
        new,
        old,
        ref,
        torch.tensor([1.0], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        clip_eps=0.2,
        kl_coef=0.0,
    )
    loss.backward()
    # d(-ratio * A)/d(logp_new) = -ratio * A when the ratio is inside the band.
    _close(
        new.grad,
        torch.tensor([[-1.05]], dtype=torch.float64),
        "gradient inside the clip band is wrong: d(-ratio * A)/d(logp_new) = -ratio * A",
    )


def test_masked_positions_contribute_nothing_even_with_extreme_values():
    new, old, ref, adv, mask = _batch(70)
    base = _call(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04)
    poisoned_new, poisoned_ref = new.clone(), ref.clone()
    poisoned_new[~mask] = 900.0  # exp(900) overflows float64: a multiply-by-mask would give NaN
    poisoned_ref[~mask] = -900.0
    out = _call(poisoned_new, old, poisoned_ref, adv, mask, clip_eps=0.2, kl_coef=0.04)
    assert torch.isfinite(out), (
        "masked-out extreme log-probs leaked into the loss: use where, not multiplication, for "
        "the mask"
    )
    expected = _reference(new, old, ref, adv, mask, 0.2, 0.04)
    _close(
        out,
        torch.tensor(expected, dtype=torch.float64),
        "masked-out positions must not change the loss: zero them out before averaging",
    )
    _close(out, base, "changing values at masked-out positions changed the loss")


def test_longer_responses_do_not_dominate_the_loss():
    # Response 0 has 1 token, response 1 has 4. A token-level global mean would weight
    # response 1 far more.
    new = torch.zeros((2, 4), dtype=torch.float64)
    old = torch.zeros((2, 4), dtype=torch.float64)
    ref = torch.zeros((2, 4), dtype=torch.float64)
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    out = _call(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.0)
    # Per response: -1 for response 0 and +1 for response 1. Averaging per response gives 0.
    _close(
        out,
        torch.tensor(0.0, dtype=torch.float64),
        "per-response averaging was not used: long responses must not outweigh short ones",
    )


def test_kl_term_is_zero_when_policy_equals_reference():
    new, old, _, adv, mask = _batch(80)
    with_kl = _call(new, old, new.clone(), adv, mask, clip_eps=0.2, kl_coef=0.5)
    without = _call(new, old, new.clone(), adv, mask, clip_eps=0.2, kl_coef=0.0)
    _close(
        with_kl,
        without,
        "the KL term is not zero when the policy equals the reference: check k3 = e^d - d - 1 "
        "with d = ref - new",
    )


def test_kl_penalty_matches_k3_estimator_reference():
    new, old, ref, adv, mask = _batch(90)
    out = _call(new, old, ref, adv, mask, clip_eps=0.3, kl_coef=0.7)
    _close(
        out,
        torch.tensor(_reference(new, old, ref, adv, mask, 0.3, 0.7), dtype=torch.float64),
        "KL-penalised loss disagrees with the reference: k3 uses e^(ref - new) - (ref - new) - 1",
    )


def test_gradient_matches_finite_differences_away_from_clip_boundaries():
    # Small log-ratios keep every ratio inside the band, so no kink sits near a sample.
    new = _randn((3, 5), 100, scale=0.2)
    old = new - _randn((3, 5), 101, scale=0.02)
    ref = new + _randn((3, 5), 102, scale=0.1)
    adv = _randn((3,), 103)
    mask = torch.ones((3, 5), dtype=torch.bool)
    mask[0, 3:] = False
    x = new.clone().requires_grad_(True)
    grpo_token_loss(x, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04).backward()

    def total(v):
        return float(grpo_token_loss(v, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04))

    h = 1e-6
    numeric = torch.zeros_like(new)
    for i in range(new.numel()):
        up, down = new.clone(), new.clone()
        up.view(-1)[i] += h
        down.view(-1)[i] -= h
        numeric.view(-1)[i] = (total(up) - total(down)) / (2 * h)
    # Central differences in float64 with h = 1e-6 agree to about 1e-9; 1e-7 is still tight.
    _close(
        x.grad,
        numeric,
        "gradient through logp_new disagrees with finite differences: check the ratio, the masks "
        "and the KL term",
        atol=1e-7,
        rtol=1e-7,
    )


def test_each_token_uses_its_own_responses_advantage():
    new = torch.zeros((2, 3), dtype=torch.float64)
    old = torch.zeros((2, 3), dtype=torch.float64)
    ref = torch.zeros((2, 3), dtype=torch.float64)
    adv = torch.tensor([2.0, -3.0], dtype=torch.float64)
    mask = torch.ones((2, 3), dtype=torch.bool)
    out = _call(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.0)
    _close(
        out,
        torch.tensor(0.5, dtype=torch.float64),
        "advantages must be applied per response: each row's tokens use that row's advantage",
    )


def test_float32_inputs_keep_dtype_and_match_reference():
    new, old, ref, adv, mask = _batch(110, dtype=torch.float32)
    out = _call(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04)
    assert out.dtype == torch.float32, "loss must stay float32 when the inputs are float32"
    _close(
        out,
        torch.tensor(_reference(new, old, ref, adv, mask, 0.2, 0.04), dtype=torch.float32),
        "float32 loss disagrees with the reference beyond rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )


def test_shape_mismatch_raises_value_error():
    new, old, ref, adv, mask = _batch(120)
    _expect_value_error(
        lambda: grpo_token_loss(new, old[:-1], ref, adv, mask),
        "logp_old with a different shape from logp_new was accepted: all log-prob grids must match",
    )
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv[:-1], mask),
        "advantages with one entry per token instead of one per response were accepted: shape "
        "must be (G,)",
    )


def test_empty_mask_row_raises_value_error():
    new, old, ref, adv, mask = _batch(130)
    mask[1] = False
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask),
        "a response with no masked-in tokens was accepted: every response needs at least one token",
    )


def test_integer_mask_raises_value_error():
    new, old, ref, adv, mask = _batch(140)
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask.to(torch.int64)),
        "an integer mask was accepted: the mask must be boolean so True means a response token",
    )


def test_non_finite_log_prob_raises_value_error():
    new, old, ref, adv, mask = _batch(150)
    new[0, 0] = float("nan")
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask),
        "a NaN log-probability was accepted: non-finite inputs must raise ValueError",
    )


@pytest.mark.parametrize(
    "clip_eps", [0.0, 1.0, -0.2], ids=["zero-clip", "one-clip", "negative-clip"]
)
def test_clip_width_outside_zero_and_one_raises_value_error(clip_eps):
    new, old, ref, adv, mask = _batch(160)
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask, clip_eps=clip_eps),
        f"clip_eps={clip_eps!r} was accepted: the clip width must lie strictly between 0 and 1",
    )


def test_negative_kl_coefficient_raises_value_error():
    new, old, ref, adv, mask = _batch(170)
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask, kl_coef=-0.1),
        "a negative KL coefficient was accepted: kl_coef must be non-negative",
    )


def test_mixed_dtypes_raise_value_error():
    new, old, ref, adv, mask = _batch(180, dtype=torch.float32)
    _expect_value_error(
        lambda: grpo_token_loss(new, old.double(), ref, adv, mask),
        "log-probabilities with mixed dtypes were accepted: require one shared floating-point "
        "dtype",
    )


def test_inputs_are_not_mutated_by_the_call():
    new, old, ref, adv, mask = _batch(190)
    before = [t.clone() for t in (new, old, ref, adv, mask)]
    _call(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.04)
    names = ("logp_new", "logp_old", "logp_ref", "advantages", "mask")
    for name, b, a in zip(names, before, (new, old, ref, adv, mask), strict=True):
        assert torch.equal(b, a), f"the {name} tensor was modified in place"


def test_default_clip_and_kl_match_the_signature_defaults():
    # The signature in problem.md fixes clip_eps=0.2 and kl_coef=0.04 as the defaults.
    new, old, ref, adv, mask = _batch(200)
    out = _call(new, old, ref, adv, mask)
    _close(
        out,
        torch.tensor(_reference(new, old, ref, adv, mask, 0.2, 0.04), dtype=torch.float64),
        "the default clip_eps or kl_coef differs from the documented values 0.2 and 0.04",
    )


def test_non_numeric_clip_eps_raises_value_error():
    new, old, ref, adv, mask = _batch(210)
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask, clip_eps="0.2"),
        "a string clip_eps was accepted or failed with the wrong error: "
        "clip_eps must be a real number",
    )


def test_non_numeric_kl_coef_raises_value_error():
    new, old, ref, adv, mask = _batch(220)
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask, kl_coef="0.04"),
        "a string kl_coef was accepted or failed with the wrong error: "
        "kl_coef must be a real number",
    )


def test_zero_responses_raise_value_error():
    empty_grid = torch.zeros((0, 3), dtype=torch.float64)
    _expect_value_error(
        lambda: grpo_token_loss(
            empty_grid,
            empty_grid,
            empty_grid,
            torch.zeros((0,), dtype=torch.float64),
            torch.zeros((0, 3), dtype=torch.bool),
        ),
        "a batch with zero responses was accepted: at least one response is required",
    )


def test_arguments_that_are_not_tensors_raise_value_error():
    new, old, ref, adv, mask = _batch(230)
    args = [new, old, ref, adv, mask]
    for position in range(len(args)):
        mixed = list(args)
        mixed[position] = mixed[position].tolist()
        _expect_value_error(
            lambda m=mixed: grpo_token_loss(*m),
            "a plain list was accepted or failed with the wrong error where a tensor is required: "
            "all five arguments must be torch tensors",
        )


def test_row_without_any_masked_in_token_raises_even_when_columns_have_one():
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    mask = torch.tensor([[True, True], [False, False]])
    _expect_value_error(
        lambda: grpo_token_loss(
            zeros, zeros, zeros, torch.tensor([1.0, -1.0], dtype=torch.float64), mask
        ),
        "a response with no masked-in tokens was accepted: every row needs at least one True",
    )


def test_non_finite_reference_log_prob_raises_value_error():
    new, old, ref, adv, mask = _batch(240)
    ref[0, 0] = float("nan")
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask),
        "a NaN reference log-probability was accepted: non-finite inputs must raise ValueError",
    )


def test_non_finite_advantage_raises_value_error():
    new, old, ref, adv, mask = _batch(250)
    adv[0] = float("inf")
    _expect_value_error(
        lambda: grpo_token_loss(new, old, ref, adv, mask),
        "an infinite advantage was accepted: every advantage must be finite",
    )


def test_clip_eps_cannot_be_passed_by_position():
    new, old, ref, adv, mask = _batch(230)
    try:
        grpo_token_loss(new, old, ref, adv, mask, 0.1)
    except TypeError:
        return
    raise AssertionError(
        "clip_eps was accepted by position: the Contract makes it keyword-only, so passing it "
        "positionally must raise TypeError"
    )
