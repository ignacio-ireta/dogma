# Hidden tests: oracle comparison with a population-std reference, sample-vs-population detection,
# all-equal groups, invariance, stability, dtype, a finite-difference gradient and validation rules.
# The oracle is used only here; the submission may not use torch.nn or the std and var routines.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import math

import pytest
import torch
from submission import group_advantages

ATOL = 1e-9
RTOL = 1e-9
# float32 keeps about 7 significant digits. Values of order one with eps = 1e-6 need about 1e-6 of
# headroom for rounding, so 1e-5 is the safe bound and still rejects a wrong divisor.
FLOAT32_ATOL = 1e-5
# A large common offset of 1e8 costs about 2e-8 absolute precision in float64 before any arithmetic.
OFFSET_ATOL = 1e-6


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(shape, seed, scale=1.0, offset=0.0, dtype=torch.float64):
    return (torch.randn(shape, generator=_gen(seed), dtype=dtype) * scale + offset).clone()


def _call(rewards, **kwargs):
    snap = rewards.clone()
    try:
        out = group_advantages(rewards, **kwargs)
    except ValueError as exc:
        pytest.fail(
            f"valid input raised ValueError ({exc}): check the contract rules", pytrace=False
        )
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes and dtypes", pytrace=False
        )
    if not torch.is_tensor(out):
        pytest.fail(
            "return a torch tensor of shape (P, G), not a list or a Python number", pytrace=False
        )
    if tuple(out.shape) != tuple(rewards.shape):
        pytest.fail(
            "advantages must have the shape of the rewards: one value per response", pytrace=False
        )
    if out.dtype != rewards.dtype:
        pytest.fail("advantages must keep the dtype of the rewards; do not downcast", pytrace=False)
    if not torch.equal(snap, rewards):
        pytest.fail("the rewards input was modified in place; build a new tensor", pytrace=False)
    return out


def _reference(rewards, eps):
    """Independent oracle: population std from the torch routine with correction 0 (divide by G)."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = torch.std(rewards, dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


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


POP_MISTAKE = (
    "advantages disagree with the population-std reference: divide the squared deviations by G, "
    "not G - 1; subtract the row mean first; add eps to the std, not inside the root"
)


def test_advantages_match_population_reference_on_random_groups():
    for seed, (prompts, group) in enumerate([(1, 2), (3, 4), (5, 8), (2, 16), (7, 3)]):
        rewards = _randn((prompts, group), seed, scale=2.0)
        out = _call(rewards)
        _close(out, _reference(rewards, 1e-6), POP_MISTAKE)


def test_sample_standard_deviation_is_not_used():
    rewards = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    out = _call(rewards)
    sample_version = torch.tensor([[-0.7071067811865476, 0.7071067811865476]], dtype=torch.float64)
    if torch.allclose(out, sample_version, atol=1e-4):
        pytest.fail(
            "advantages use the sample std (divide by G - 1): use the population std (divide by G)",
            pytrace=False,
        )
    _close(out, _reference(rewards, 1e-6), POP_MISTAKE)


def test_eps_is_added_to_the_std_outside_the_square_root():
    # rewards [0, 1]: mean 0.5, population variance 0.25. With eps = 0.5 the correct divisor is
    # std + eps = 0.5 + 0.5 = 1.0, so the advantage is -0.5. Putting eps inside the root gives
    # sqrt(0.25 + 0.5) = 0.866 instead, so the advantage is about -0.577.
    rewards = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    out = _call(rewards, eps=0.5)
    expected = torch.tensor([[-0.5, 0.5]], dtype=torch.float64)
    _close(
        out,
        expected,
        "eps was added inside the square root: add eps to the std, after the root",
        atol=1e-12,
        rtol=1e-12,
    )


def test_all_equal_group_gives_finite_zero_advantages_across_magnitudes():
    for value in (0.0, 1e-6, 0.7, 1e6, -3.0):
        rewards = torch.full((2, 4), value, dtype=torch.float64)
        out = _call(rewards)
        assert torch.isfinite(out).all(), (
            "all-equal group produced NaN or inf: the std is zero, "
            "so eps must keep the divisor positive"
        )
        _close(
            out,
            torch.zeros_like(out),
            "all-equal group must give zero advantages, not a spread",
            atol=1e-9,
        )


def test_all_equal_rows_do_not_affect_other_groups():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    out = _call(rewards)
    _close(
        out[0],
        _reference(rewards, 1e-6)[0],
        "a group's advantages depend on other groups: normalise each row alone",
    )
    _close(
        out[1], torch.zeros(4, dtype=torch.float64), "an all-equal group must give zeros", atol=1e-9
    )


def test_each_group_advantage_sums_to_zero():
    rewards = _randn((6, 5), 11, scale=3.0)
    out = _call(rewards)
    row_sums = out.sum(dim=1)
    _close(
        row_sums,
        torch.zeros(6, dtype=torch.float64),
        "advantages within a group must sum to zero: subtract the group mean",
        atol=1e-9,
    )


def test_adding_a_constant_to_one_group_changes_nothing():
    rewards = _randn((3, 4), 21, scale=1.0)
    base = _call(rewards)
    shifted = rewards.clone()
    shifted[1] += 42.0
    out = _call(shifted)
    _close(
        out[1],
        base[1],
        "adding a constant to a group changed its advantages: subtract the group mean",
    )
    _close(out[0], base[0], "shifting one group changed another group's advantages")


def test_large_common_offset_keeps_the_spread_exact():
    rewards = _randn((4, 6), 31, scale=1.0, offset=1e8)
    out = _call(rewards)
    centred = _randn((4, 6), 31, scale=1.0)
    _close(
        out,
        _reference(centred, 1e-6),
        "large-offset advantages disagree with the reference: "
        "compute variance from centred values, "
        "not from E[x^2] - E[x]^2, which cancels at large offsets",
        atol=OFFSET_ATOL,
        rtol=OFFSET_ATOL,
    )


def test_scaling_all_rewards_keeps_the_advantage_for_a_tiny_eps():
    rewards = _randn((3, 5), 41, scale=1.0)
    base = _call(rewards, eps=1e-12)
    scaled = _call(rewards * 250.0, eps=1e-12)
    _close(
        scaled,
        base,
        "advantages are not scale-free: the std must scale with the rewards",
        atol=1e-6,
        rtol=1e-6,
    )


def test_float32_inputs_keep_dtype_and_match_reference():
    rewards = _randn((4, 8), 51, scale=2.0, dtype=torch.float32)
    out = _call(rewards)
    assert out.dtype == torch.float32, "advantages must stay float32 when rewards are float32"
    _close(
        out,
        _reference(rewards, 1e-6),
        "float32 advantages disagree with the reference beyond rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )


def test_gradient_matches_finite_differences_in_float64():
    rewards = _randn((2, 4), 61, scale=1.0)
    weights = _randn((2, 4), 62, scale=1.0)
    x = rewards.clone().requires_grad_(True)
    (group_advantages(x) * weights).sum().backward()
    h = 1e-6
    numeric = torch.zeros_like(rewards)
    for i in range(rewards.numel()):
        up, down = rewards.clone(), rewards.clone()
        up.view(-1)[i] += h
        down.view(-1)[i] -= h
        plus = float((group_advantages(up) * weights).sum())
        minus = float((group_advantages(down) * weights).sum())
        numeric.view(-1)[i] = (plus - minus) / (2 * h)
    # Central differences in float64 with h = 1e-6 agree to about 1e-9; 1e-7 is still tight.
    _close(
        x.grad,
        numeric,
        "gradient through the advantages disagrees with finite differences: "
        "check the mean and the std",
        atol=1e-7,
        rtol=1e-7,
    )


def test_one_response_group_raises_value_error():
    _expect_value_error(
        lambda: group_advantages(torch.tensor([[1.0]], dtype=torch.float64)),
        "a group of one response was accepted: a group needs at least two responses",
    )


def test_zero_prompts_raise_value_error():
    _expect_value_error(
        lambda: group_advantages(torch.zeros((0, 4), dtype=torch.float64)),
        "an input with no prompts was accepted: at least one group is required",
    )


def test_one_dimensional_rewards_raise_value_error():
    _expect_value_error(
        lambda: group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64)),
        "1-D rewards were accepted: pass one row per prompt, shape (P, G)",
    )


def test_nan_reward_raises_value_error():
    rewards = torch.tensor([[1.0, float("nan")]], dtype=torch.float64)
    _expect_value_error(
        lambda: group_advantages(rewards),
        "a NaN reward was accepted: non-finite rewards must raise ValueError",
    )


def test_integer_rewards_raise_value_error():
    _expect_value_error(
        lambda: group_advantages(torch.tensor([[1, 0, 0, 1]])),
        "integer rewards were accepted: the advantage needs floating-point rewards",
    )


@pytest.mark.parametrize("eps", [0.0, -1e-6, math.nan])
def test_non_positive_eps_raises_value_error(eps):
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    _expect_value_error(
        lambda: group_advantages(rewards, eps=eps),
        "eps must be finite and strictly positive: a zero eps turns an all-equal group into 0/0",
    )


def test_rewards_that_are_not_a_tensor_raise_value_error():
    _expect_value_error(
        lambda: group_advantages([[1.0, 0.0, 0.0, 1.0]]),
        "rewards given as a plain list was accepted or failed with the wrong error: "
        "rewards must be a torch tensor",
    )


def test_eps_that_is_not_a_number_raises_value_error():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    _expect_value_error(
        lambda: group_advantages(rewards, eps="0.1"),
        "a string eps was accepted or failed with the wrong error: "
        "eps must be a finite positive number",
    )
