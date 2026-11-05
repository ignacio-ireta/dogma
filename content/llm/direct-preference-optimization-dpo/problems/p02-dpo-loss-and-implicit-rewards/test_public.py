# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import torch
from submission import dpo_loss_and_implicit_rewards


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_implicit_rewards_and_loss_for_a_pair_the_policy_prefers():
    """beta=0.1 gives rewards 0.1 and -0.1, margin 0.2, loss -log(sigmoid(0.2))."""
    loss, r_c, r_r = dpo_loss_and_implicit_rewards(
        _t([-10.0]), _t([-12.0]), _t([-11.0]), _t([-11.0]), beta=0.1
    )
    assert torch.allclose(r_c, _t([0.1]), atol=1e-12), (
        "chosen implicit reward is wrong: beta * (policy log-prob - reference log-prob)"
    )
    assert torch.allclose(r_r, _t([-0.1]), atol=1e-12), (
        "rejected implicit reward is wrong: use the rejected response's own log-probs"
    )
    assert torch.allclose(loss, _t(0.5981388693815918), atol=1e-9), (
        "loss is wrong: the DPO loss is -log(sigmoid(r_chosen - r_rejected)), averaged over pairs"
    )


def test_policy_equal_to_reference_gives_log_two_and_zero_rewards():
    """Policy equal to reference: both implicit rewards are zero and the loss is log 2."""
    loss, r_c, r_r = dpo_loss_and_implicit_rewards(
        _t([-3.0]), _t([-5.0]), _t([-3.0]), _t([-5.0]), beta=0.1
    )
    assert torch.allclose(r_c, _t([0.0]), atol=1e-12) and torch.allclose(
        r_r, _t([0.0]), atol=1e-12
    ), "implicit rewards must be zero when the policy equals the reference"
    assert torch.allclose(loss, _t(math.log(2.0)), atol=1e-9), (
        "a zero margin must give log 2: check that the reference is subtracted"
    )


def test_non_log_probability_input_raises_value_error():
    """A positive value is a probability, not a log-probability, and must raise ValueError."""
    try:
        dpo_loss_and_implicit_rewards(_t([0.5]), _t([-1.0]), _t([-1.0]), _t([-1.0]), beta=0.1)
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"a probability passed as a log-probability raised {type(exc).__name__}: use ValueError"
        ) from None
    raise AssertionError("a probability was accepted as a log-probability: values must be <= 0")
