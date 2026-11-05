from __future__ import annotations

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only: the submission may not use torch.nn
from submission import reinforce_loss

ATOL = 1e-9
RTOL = 1e-9
FD_STEP = 1e-6
FD_TOL = 1e-6


def _call(*args, **kwargs):
    try:
        return reinforce_loss(*args, **kwargs)
    except Exception as exc:  # a valid call must not fail, whatever the exception type
        pytest.fail(f"valid input raised {type(exc).__name__}: check shapes, dtypes and the loss")


def _rejects(*args, **kwargs):
    try:
        reinforce_loss(*args, **kwargs)
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception type")
    pytest.fail("invalid input was accepted: check the argument validation")


def _gradient(loss, z):
    """Backpropagate and return the gradient with respect to z, naming any broken link first."""
    if not loss.requires_grad:
        pytest.fail(
            "the loss does not depend on the logits: check the log-probabilities are not detached",
            pytrace=False,
        )
    loss.backward()
    if z.grad is None:
        pytest.fail(
            "no gradient reached the logits: check the graph from logits to the loss", pytrace=False
        )
    return z.grad


def _returns(rewards, gamma):
    """Independent oracle: the defining double sum for reward-to-go, one episode at a time."""
    batch, steps = rewards.shape
    out = torch.zeros_like(rewards)
    for b in range(batch):
        for t in range(steps):
            out[b, t] = sum((gamma**k) * rewards[b, t + k] for k in range(steps - t))
    return out


def _reference_loss(logits, actions, rewards, gamma, baseline):
    """Independent oracle: cross-entropy per step, weighted by the advantage, averaged over B."""
    returns = _returns(rewards, gamma)
    adv = returns - returns.mean(dim=0, keepdim=True) if baseline else returns
    batch, steps, n_actions = logits.shape
    nll = ref_f.cross_entropy(
        logits.reshape(-1, n_actions), actions.reshape(-1), reduction="none"
    ).reshape(batch, steps)
    return (nll * adv).sum() / batch


def _random_batch(seed, batch, steps, n_actions, dtype=torch.float64, scale=1.0):  # noqa: PLR0917 (the contract fixes these arguments)
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, steps, n_actions, generator=g, dtype=dtype) * scale
    actions = torch.randint(0, n_actions, (batch, steps), generator=g)
    rewards = torch.randn(batch, steps, generator=g, dtype=dtype)
    return logits, actions, rewards


def _closed_form_grad(logits, actions, rewards, gamma, baseline):
    """Score-function gradient with respect to the logits, written out without autograd."""
    returns = _returns(rewards, gamma)
    adv = returns - returns.mean(dim=0, keepdim=True) if baseline else returns
    probs = torch.softmax(logits, dim=-1)
    onehot = torch.nn.functional.one_hot(actions, logits.shape[-1]).to(logits.dtype)
    batch = logits.shape[0]
    return -(adv.unsqueeze(-1) * (onehot - probs)) / batch


def test_loss_matches_cross_entropy_reference_on_random_batches():
    for seed in range(12):
        batch, steps, n_actions = 2 + seed % 3, 1 + seed % 5, 2 + seed % 4
        gamma = (0.0, 0.5, 0.9, 1.0)[seed % 4]
        for baseline in (False, True):
            logits, actions, rewards = _random_batch(seed, batch, steps, n_actions)
            got = _call(logits, actions, rewards, gamma, baseline)
            want = _reference_loss(logits, actions, rewards, gamma, baseline)
            assert got.shape == torch.Size([]), "the loss must be a 0-dimensional tensor"
            assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
                "loss differs from the cross-entropy reference: check the reward-to-go, the "
                "baseline (mean over episodes at each step) and the division by the batch size"
            )


def test_logit_gradient_matches_closed_form_score_function():
    for seed in range(8):
        logits, actions, rewards = _random_batch(100 + seed, 3, 4, 3)
        logits.requires_grad_(True)
        loss = _call(logits, actions, rewards, 0.9, seed % 2 == 0)
        grad = _gradient(loss, logits)
        want = _closed_form_grad(logits.detach(), actions, rewards, 0.9, seed % 2 == 0)
        assert torch.allclose(grad, want, atol=ATOL, rtol=RTOL), (
            "logit gradient differs from the score function: check the sign and the "
            "one-hot minus softmax term"
        )


def test_logit_gradient_matches_finite_differences_in_float64():
    logits, actions, rewards = _random_batch(7, 2, 3, 3)
    gamma = 0.8

    def objective(z):
        return float(_call(z, actions, rewards, gamma, True))

    numeric = torch.zeros_like(logits)
    flat = logits.reshape(-1)
    for index in range(flat.numel()):
        old = flat[index].item()
        flat[index] = old + FD_STEP
        plus = objective(logits)
        flat[index] = old - FD_STEP
        minus = objective(logits)
        flat[index] = old
        numeric.reshape(-1)[index] = (plus - minus) / (2 * FD_STEP)

    z = logits.clone().requires_grad_(True)
    grad = _gradient(_call(z, actions, rewards, gamma, True), z)
    assert torch.allclose(grad, numeric, atol=FD_TOL, rtol=FD_TOL), (
        "autograd gradient disagrees with central differences: check the loss is differentiable "
        "in the logits and no step is detached"
    )


def test_constant_returns_with_baseline_give_zero_loss_and_gradient():
    steps, batch, n_actions = 3, 4, 3
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(batch, steps, n_actions, generator=g, dtype=torch.float64)
    actions = torch.randint(0, n_actions, (batch, steps), generator=g)
    rewards = torch.ones(batch, steps, dtype=torch.float64)  # identical episodes, identical returns
    z = logits.clone().requires_grad_(True)
    loss = _call(z, actions, rewards, 1.0, True)
    assert abs(float(loss)) < ATOL, "identical episodes with a mean baseline must give zero loss"
    grad = _gradient(loss, z)
    assert torch.allclose(grad, torch.zeros_like(z), atol=ATOL), (
        "identical episodes with a mean baseline must give zero gradient"
    )


def test_gamma_zero_uses_only_the_immediate_reward():
    logits, actions, rewards = _random_batch(21, 3, 4, 2)
    got = _call(logits, actions, rewards, 0.0, False)
    want = _reference_loss(logits, actions, rewards, 0.0, False)
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
        "with gamma = 0 the weight must be the immediate reward r[b, t], not a later return"
    )


def test_gamma_one_sums_rewards_to_the_end_of_the_episode():
    logits, actions, rewards = _random_batch(22, 2, 5, 3)
    got = _call(logits, actions, rewards, 1.0, False)
    suffix = torch.flip(torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    want = -(log_probs * suffix).sum() / rewards.shape[0]
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
        "with gamma = 1 the weight must be the suffix sum of rewards from step t"
    )


def test_extreme_logits_stay_finite_and_match_the_reference():
    logits, _, rewards = _random_batch(31, 2, 3, 4, scale=1e3)
    # The least likely action at each step has log-probability near -2000, where a naive
    # log(softmax) underflows to log(0) = -inf. Sampled actions must include those.
    actions = logits.argmin(dim=-1)
    got = _call(logits, actions, rewards, 0.9, True)
    assert bool(torch.isfinite(got)), (
        "loss is not finite for large logits: compute log-softmax with the maximum subtracted"
    )
    want = _reference_loss(logits, actions, rewards, 0.9, True)
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
        "large-logit loss differs from the reference: log-softmax must be numerically stable"
    )


def test_float32_inputs_return_float32_close_to_the_float64_reference():
    logits, actions, rewards = _random_batch(41, 2, 3, 3, dtype=torch.float32)
    got = _call(logits, actions, rewards, 0.9, False)
    assert got.dtype == torch.float32, "the loss must keep the dtype of the logits"
    want = _reference_loss(logits.double(), actions, rewards.double(), 0.9, False)
    # float32 carries about seven significant digits, so this reference is compared at 1e-4.
    assert abs(float(got) - float(want)) <= 1e-4 * max(1.0, abs(float(want))), (
        "float32 loss is too far from the float64 reference: check the dtype handling"
    )


def test_float64_rewards_with_float32_logits_still_return_float32():
    # The contract ties the result to the dtype of logits, whatever the dtype of rewards is.
    logits, actions, rewards = _random_batch(42, 2, 3, 3, dtype=torch.float32)
    got = _call(logits, actions, rewards.double(), 0.9, True)
    assert got.dtype == torch.float32, "the loss must have the dtype of the logits, not of rewards"


def test_single_action_gives_zero_loss_and_gradient():
    logits = torch.randn(2, 3, 1, dtype=torch.float64, requires_grad=True)
    actions = torch.zeros(2, 3, dtype=torch.long)
    rewards = torch.randn(2, 3, dtype=torch.float64)
    loss = _call(logits, actions, rewards, 0.9, True)
    assert abs(float(loss)) < ATOL, "with one action log pi = 0 everywhere, so the loss is zero"
    grad = _gradient(loss, logits)
    assert torch.allclose(grad, torch.zeros_like(logits), atol=ATOL), (
        "with one action the score function is zero, so the gradient must be zero"
    )


def test_loss_keeps_the_graph_to_the_logits():
    logits, actions, rewards = _random_batch(51, 2, 2, 3)
    logits.requires_grad_(True)
    loss = _call(logits, actions, rewards, 0.9, False)
    grad = _gradient(loss, logits)
    assert bool(torch.any(grad != 0)), (
        "no gradient reached the logits: check that the log-probabilities are not detached"
    )


def test_inputs_are_not_mutated():
    logits, actions, rewards = _random_batch(61, 2, 3, 3)
    originals = [logits.clone(), actions.clone(), rewards.clone()]
    _call(logits, actions, rewards, 0.7, True)
    for got, want in zip((logits, actions, rewards), originals, strict=True):
        assert torch.equal(got, want), "an input tensor was modified"


def test_invalid_arguments_are_rejected():
    logits, actions, rewards = _random_batch(71, 2, 3, 3)
    _rejects(logits[0], actions, rewards, 0.9)  # two dimensions instead of three
    _rejects(logits, actions.float(), rewards, 0.9)  # float actions
    _rejects(logits, actions, rewards[:, :2], 0.9)  # reward shape mismatch
    _rejects(logits, actions + 3, rewards, 0.9)  # action index out of range
    _rejects(logits, actions, torch.full_like(rewards, float("nan")), 0.9)
    _rejects(logits, actions, rewards, -0.1)
    _rejects(logits, actions, rewards, 1.1)
    _rejects(logits, actions, rewards, float("nan"))
    _rejects(logits, actions, rewards, True)
    _rejects(logits, actions, rewards, 0.9, 1)  # baseline must be a bool
    _rejects(logits[:0], actions[:0], rewards[:0], 0.9)  # empty batch


def test_empty_episodes_are_rejected():
    _rejects(torch.zeros(2, 0, 3), torch.zeros(2, 0, dtype=torch.long), torch.zeros(2, 0), 0.9)


def test_bool_actions_are_rejected():
    logits, _, rewards = _random_batch(72, 2, 3, 3)
    _rejects(logits, torch.ones(2, 3, dtype=torch.bool), rewards, 0.9)


def test_integer_logits_are_rejected_as_not_floating_point():
    logits = torch.zeros(2, 3, 3, dtype=torch.long)
    _rejects(
        logits, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.float64), 0.9
    )


def test_actions_with_the_wrong_episode_length_are_rejected():
    logits, actions, rewards = _random_batch(73, 2, 3, 3)
    _rejects(logits, actions[:, :1], rewards, 0.9)


def test_integer_rewards_are_rejected_as_not_floating_point():
    logits, actions, _ = _random_batch(74, 2, 3, 3)
    _rejects(logits, actions, torch.zeros(2, 3, dtype=torch.long), 0.9)


def test_rewards_with_one_nan_entry_are_rejected():
    logits, actions, rewards = _random_batch(75, 2, 3, 3)
    rewards = rewards.clone()
    rewards[0, 1] = float("nan")
    _rejects(logits, actions, rewards, 0.9)


def test_negative_action_index_is_rejected():
    logits, actions, rewards = _random_batch(76, 2, 3, 3)
    actions = actions.clone()
    actions[0, 0] = -1
    _rejects(logits, actions, rewards, 0.9)


def test_action_index_equal_to_the_action_count_is_rejected():
    logits, actions, rewards = _random_batch(77, 2, 3, 3)
    actions = actions.clone()
    actions[0, 0] = 0
    actions[0, 1] = 3  # the valid indices are 0, 1 and 2
    _rejects(logits, actions, rewards, 0.9)
