# Hidden tests: oracle comparison with PyTorch's log-sigmoid, reward scaling and cancellation,
# the analytic gradient and a finite-difference check, stability, dtype and validation rules.
# The oracle is used only here; the submission may not use torch.nn or its functional module.
# Messages name the kind of mistake and never show hidden inputs or expected values.
from __future__ import annotations

import collections
import functools
import math
import re

import pytest
import torch
import torch.nn.functional as ref_f  # oracle only
from submission import dpo_loss_and_implicit_rewards

from dogma_data import load_text
from dogma_data import rng as data_rng

ATOL = 1e-9
RTOL = 1e-9
# float32 keeps about 7 significant digits. Log-probs near -10 carry about 1e-6 absolute rounding,
# and beta scales them, so 1e-5 leaves room for rounding and still rejects a wrong formula.
FLOAT32_ATOL = 1e-5


@functools.cache
def _language_models():
    """Two real unigram language models and the responses they score.

    The policy is the word frequencies of Frankenstein and the reference those of Walden, both
    with one added count per word of the joint vocabulary. A response is the first three words
    of a line of Pride and Prejudice, and its log-probability under a model is the sum of the
    log-probabilities of its words: strictly negative, spread over a few nats.
    """
    words = {
        name: re.findall(r"[a-z]+", load_text(name).lower())
        for name in ("gutenberg_frankenstein", "gutenberg_walden", "gutenberg_pride_prejudice")
    }
    vocab = sorted(set().union(*words.values()))
    tables = []
    for name in ("gutenberg_frankenstein", "gutenberg_walden"):
        counts = collections.Counter(words[name])
        total = len(words[name]) + len(vocab)
        tables.append({word: math.log((counts[word] + 1) / total) for word in vocab})
    responses = [
        tuple(re.findall(r"[a-z]+", line.lower())[:3])
        for line in load_text("gutenberg_pride_prejudice").splitlines()
    ]
    responses = [response for response in responses if len(response) == 3]
    scores = [[sum(table[word] for word in response) for response in responses] for table in tables]
    return torch.tensor(scores, dtype=torch.float64)


def _inputs(seed, batch=8, dtype=torch.float64):
    """Policy and reference log-probabilities of real chosen and rejected responses."""
    scores = _language_models()
    picked = data_rng(f"p02-dpo-loss-and-implicit-rewards:responses:{seed}").choice(
        scores.shape[1], size=2 * batch, replace=False
    )
    chosen, rejected = torch.tensor(picked[:batch]), torch.tensor(picked[batch:])
    policy, reference = scores[0].to(dtype), scores[1].to(dtype)
    return (
        policy[chosen].clone(),
        policy[rejected].clone(),
        reference[chosen].clone(),
        reference[rejected].clone(),
    )


def _call(pc, pr, rc, rr, beta):
    snap = [t.clone() for t in (pc, pr, rc, rr)]
    try:
        out = dpo_loss_and_implicit_rewards(pc, pr, rc, rr, beta=beta)
    except ValueError as exc:
        pytest.fail(
            f"valid input raised ValueError ({exc}): check the contract rules", pytrace=False
        )
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check shapes, dtypes and the contract",
            pytrace=False,
        )
    if not isinstance(out, tuple) or len(out) != 3 or not all(torch.is_tensor(t) for t in out):
        pytest.fail(
            "return a tuple of three tensors: (loss, chosen_reward, rejected_reward)", pytrace=False
        )
    loss, r_c, r_r = out
    if loss.ndim != 0:
        pytest.fail("loss must be a 0-dimensional tensor: average over the pairs", pytrace=False)
    if r_c.shape != pc.shape or r_r.shape != pc.shape:
        pytest.fail("implicit rewards must have shape (B,), one per pair", pytrace=False)
    if loss.dtype != pc.dtype or r_c.dtype != pc.dtype or r_r.dtype != pc.dtype:
        pytest.fail("loss and rewards must keep the input dtype; do not downcast", pytrace=False)
    for before, after in zip(snap, (pc, pr, rc, rr), strict=True):
        if not torch.equal(before, after):
            pytest.fail("an input tensor was modified in place; build new tensors", pytrace=False)
    return loss, r_c, r_r


def _reference(pc, pr, rc, rr, beta):
    """Independent oracle: rewards as beta-scaled log-ratios, loss from PyTorch's log-sigmoid."""
    r_c = beta * (pc - rc)
    r_r = beta * (pr - rr)
    loss = -ref_f.logsigmoid(r_c - r_r).mean()
    return loss, r_c, r_r


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


def test_loss_and_rewards_match_reference_over_several_betas():
    for seed, beta in enumerate([0.05, 0.1, 0.5, 2.0]):
        pc, pr, rc, rr = _inputs(seed * 10)
        loss, r_c, r_r = _call(pc, pr, rc, rr, beta)
        want_loss, want_c, want_r = _reference(pc, pr, rc, rr, beta)
        _close(
            r_c,
            want_c,
            "chosen implicit reward is wrong: "
            "use beta * (policy - reference) for the chosen response",
        )
        _close(
            r_r,
            want_r,
            "rejected implicit reward is wrong: use the rejected response's own log-probs",
        )
        _close(
            loss,
            want_loss,
            "DPO loss disagrees with the reference: "
            "check the margin is chosen reward minus rejected "
            "reward, the sign of -log(sigmoid), and that the loss is averaged over pairs",
        )


def test_implicit_rewards_scale_linearly_with_beta():
    pc, pr, rc, rr = _inputs(200)
    _, c1, r1 = _call(pc, pr, rc, rr, 1.0)
    _, c3, r3 = _call(pc, pr, rc, rr, 3.0)
    _close(
        c3,
        3.0 * c1,
        "chosen reward does not scale linearly with beta: beta must multiply the log-ratio",
    )
    _close(
        r3,
        3.0 * r1,
        "rejected reward does not scale linearly with beta: beta must multiply the log-ratio",
    )


def test_shared_constant_on_policy_and_reference_cancels():
    pc, pr, rc, rr = _inputs(300)
    base = _call(pc, pr, rc, rr, 0.3)
    shifted = _call(pc - 5.0, pr, rc - 5.0, rr, 0.3)
    for label, a, b in zip(
        ("loss", "chosen reward", "rejected reward"), base, shifted, strict=True
    ):
        _close(
            b,
            a,
            f"{label} changed when a constant was added to the policy and reference "
            "chosen log-probs: "
            "the implicit reward must use their difference",
        )


def test_policy_equal_to_reference_gives_log_two_and_zero_rewards():
    pc, pr, _, _ = _inputs(400)
    loss, r_c, r_r = _call(pc, pr, pc.clone(), pr.clone(), 0.25)
    _close(r_c, torch.zeros_like(pc), "rewards must be zero when the policy equals the reference")
    _close(r_r, torch.zeros_like(pc), "rewards must be zero when the policy equals the reference")
    _close(
        loss,
        torch.tensor(float(torch.log(torch.tensor(2.0, dtype=torch.float64))), dtype=torch.float64),
        "a zero margin must give log 2",
    )


def test_large_margins_of_either_sign_stay_finite_and_exact():
    # With beta = 1, pair 0 has margin about +2e4 and pair 1 about -2e4; all log-probs stay <= 0.
    pc = torch.tensor([-1.0, -1.0e4], dtype=torch.float64)
    pr = torch.tensor([-1.0e4, -1.0], dtype=torch.float64)
    rc = torch.tensor([-1.0e4, -1.0], dtype=torch.float64)
    rr = torch.tensor([-1.0, -1.0e4], dtype=torch.float64)
    loss, _, _ = _call(pc, pr, rc, rr, 1.0)
    assert torch.isfinite(loss), (
        "loss overflowed for large margins: use the softplus form of -log(sigmoid)"
    )
    want_loss, _, _ = _reference(pc, pr, rc, rr, 1.0)
    _close(
        loss, want_loss, "loss for margins of 1e4 disagrees with the stable reference", rtol=1e-9
    )


def test_float32_inputs_keep_dtype_and_match_reference():
    pc, pr, rc, rr = _inputs(500, dtype=torch.float32)
    loss, r_c, r_r = _call(pc, pr, rc, rr, 0.1)
    assert loss.dtype == torch.float32, "loss must stay float32 when the inputs are float32"
    want_loss, want_c, want_r = _reference(pc, pr, rc, rr, 0.1)
    _close(
        r_c,
        want_c,
        "float32 chosen reward disagrees with the reference beyond rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )
    _close(
        r_r,
        want_r,
        "float32 rejected reward disagrees with the reference beyond rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )
    _close(
        loss,
        want_loss,
        "float32 loss disagrees with the reference beyond rounding",
        atol=FLOAT32_ATOL,
        rtol=FLOAT32_ATOL,
    )


def test_gradient_wrt_policy_matches_analytic_form_in_float64():
    beta = 0.2
    pc, pr, rc, rr = _inputs(600)
    pc = pc.clone().requires_grad_(True)
    pr = pr.clone().requires_grad_(True)
    loss, _, _ = dpo_loss_and_implicit_rewards(pc, pr, rc, rr, beta=beta)
    loss.backward()
    margin = beta * ((pc - rc) - (pr - rr)).detach()
    # d(softplus(-m))/dm = -sigmoid(-m); m depends on the policy log-probs with coefficient beta,
    # and the mean divides by B.
    coeff = -beta * torch.sigmoid(-margin) / pc.numel()
    _close(
        pc.grad,
        coeff,
        "gradient wrt the chosen policy log-prob disagrees with -beta * sigmoid(-margin) / B: "
        "check the sign and the beta factor",
    )
    _close(
        pr.grad,
        -coeff,
        "gradient wrt the rejected policy log-prob disagrees with +beta * sigmoid(-margin) / B: "
        "check the sign",
    )


def test_gradient_matches_central_finite_differences_in_float64():
    beta = 0.7
    pc, pr, rc, rr = _inputs(700, batch=4)
    x = pc.clone().requires_grad_(True)
    loss, _, _ = dpo_loss_and_implicit_rewards(x, pr, rc, rr, beta=beta)
    loss.backward()

    def total(v):
        value, _, _ = dpo_loss_and_implicit_rewards(v, pr, rc, rr, beta=beta)
        return float(value)

    h = 1e-6
    numeric = torch.zeros_like(pc)
    for i in range(pc.numel()):
        up, down = pc.clone(), pc.clone()
        up[i] += h
        down[i] -= h
        numeric[i] = (total(up) - total(down)) / (2 * h)
    # Central differences in float64 with h = 1e-6 agree to about 1e-9; 1e-7 is still tight.
    _close(
        x.grad,
        numeric,
        "autograd gradient disagrees with finite differences: the loss may be detached",
        atol=1e-7,
        rtol=1e-7,
    )


def test_probability_passed_as_log_probability_raises_value_error():
    pc, pr, rc, rr = _inputs(800)
    pc = pc.clone()
    pc[0] = 0.4  # a probability, not a log-probability
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(pc, pr, rc, rr, beta=0.1),
        "a positive value was accepted as a log-probability: values must be <= 0",
    )


def test_non_positive_beta_raises_value_error():
    pc, pr, rc, rr = _inputs(900)
    for beta in (0.0, -0.1, float("nan"), float("inf")):
        _expect_value_error(
            lambda beta=beta: dpo_loss_and_implicit_rewards(pc, pr, rc, rr, beta=beta),
            f"beta={beta!r} was accepted: beta must be finite and strictly positive",
        )


def test_shape_mismatch_raises_value_error():
    pc, pr, rc, rr = _inputs(910)
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(pc, pr, rc[:-1], rr, beta=0.1),
        "reference log-probs with a different length were accepted: "
        "all four must have one entry per pair",
    )


def test_empty_batch_raises_value_error():
    empty = torch.zeros((0,), dtype=torch.float64)
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(empty, empty, empty, empty, beta=0.1),
        "an empty batch was accepted: at least one pair is required",
    )


def test_two_dimensional_inputs_raise_value_error():
    pc = torch.full((2, 3), -1.0, dtype=torch.float64)
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(pc, pc, pc, pc, beta=0.1),
        "2-D log-probs were accepted: pass one value per pair as a 1-D tensor",
    )


def test_nan_log_probability_raises_value_error():
    pc, pr, rc, rr = _inputs(920)
    pc = pc.clone()
    pc[1] = float("nan")
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(pc, pr, rc, rr, beta=0.1),
        "a NaN log-probability was accepted: non-finite values must raise ValueError",
    )


def test_integer_log_probabilities_raise_value_error():
    ints = torch.full((3,), -2, dtype=torch.int64)
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(ints, ints, ints, ints, beta=0.1),
        "integer log-probabilities were accepted: they must be floating-point",
    )


def test_mixed_dtypes_raise_value_error():
    pc, pr, rc, rr = _inputs(930, dtype=torch.float32)
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(pc, pr, rc.double(), rr, beta=0.1),
        "log-probabilities with mixed dtypes were accepted: require one shared dtype",
    )


def test_inputs_are_not_mutated_by_the_call():
    pc, pr, rc, rr = _inputs(940)
    before = [t.clone() for t in (pc, pr, rc, rr)]
    _call(pc, pr, rc, rr, 0.4)
    for name, b, a in zip(
        ("policy chosen", "policy rejected", "ref chosen", "ref rejected"),
        before,
        (pc, pr, rc, rr),
        strict=True,
    ):
        assert torch.equal(b, a), f"the {name} tensor was modified in place"


def test_list_in_place_of_a_tensor_raises_value_error():
    pc, pr, rc, rr = _inputs(960)
    args = [pc, pr, rc, rr]
    for position in range(len(args)):
        mixed = list(args)
        mixed[position] = mixed[position].tolist()
        _expect_value_error(
            lambda m=mixed: dpo_loss_and_implicit_rewards(*m, beta=0.1),
            "a plain list was accepted or failed with the wrong error where a tensor is required: "
            "every log-probability argument must be a torch tensor",
        )


def test_log_probability_of_exactly_zero_is_accepted():
    pc, pr, rc, rr = _inputs(950)
    pc = pc.clone()
    pc[0] = 0.0  # a response with probability one has log-probability exactly 0
    loss, r_c, _r_r = _call(pc, pr, rc, rr, 0.3)
    want_loss, want_c, _want_r = _reference(pc, pr, rc, rr, 0.3)
    _close(
        r_c,
        want_c,
        "a log-probability of exactly 0 was mishandled in the chosen reward: "
        "the contract allows values up to and including 0",
    )
    _close(
        loss,
        want_loss,
        "a log-probability of exactly 0 was mishandled in the loss: "
        "the contract allows values up to and including 0",
    )


def test_beta_that_is_not_a_number_raises_value_error():
    pc, pr, rc, rr = _inputs(970)
    _expect_value_error(
        lambda: dpo_loss_and_implicit_rewards(pc, pr, rc, rr, beta="0.1"),
        "a string beta was accepted or failed with the wrong error: "
        "beta must be a finite positive number",
    )
