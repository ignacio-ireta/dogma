# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import grpo_token_loss


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _call(*args, **kwargs):
    before = [tensor.detach().clone() for tensor in args]
    try:
        loss = grpo_token_loss(*args, **kwargs)
    except RuntimeError:
        pytest.fail(
            "the valid loss inputs raised a tensor error: check shapes and avoid in-place "
            "updates to tensors used by autograd",
            pytrace=False,
        )
    names = ("logp_new", "logp_old", "logp_ref", "advantages", "mask")
    for name, original, current in zip(names, before, args, strict=True):
        assert torch.equal(original, current), (
            f"the {name} input was modified in place: compute with new tensors instead"
        )
    return loss


def test_single_unclipped_token_gives_negative_advantage():
    """Ratio 1, advantage 1 and no KL: the loss is -1."""
    loss = _call(
        _t([[0.0]]), _t([[0.0]]), _t([[0.0]]), _t([1.0]), torch.tensor([[True]]), kl_coef=0.0
    )
    assert loss.ndim == 0, "loss must be a 0-dimensional tensor"
    assert torch.allclose(loss, _t(-1.0), atol=1e-9), (
        "single-token loss is wrong: the surrogate is ratio * advantage, and the loss is its "
        "negative"
    )


def test_positive_advantage_above_clip_is_clipped_at_one_plus_eps():
    """Ratio 1.5 with advantage 1 and eps 0.2 is clipped to 1.2, so the loss is -1.2."""
    loss = _call(
        _t([[math.log(1.5)]]),
        _t([[0.0]]),
        _t([[math.log(1.5)]]),
        _t([1.0]),
        torch.tensor([[True]]),
        clip_eps=0.2,
        kl_coef=0.0,
    )
    assert torch.allclose(loss, _t(-1.2), atol=1e-9), (
        "clipping is wrong: for a positive advantage the surrogate is min(ratio, 1 + eps) * "
        "advantage"
    )


def test_mean_over_two_opposite_advantages_is_zero():
    """Advantages +1 and -1 with ratios 1 and no KL: the mean of -1 and +1 is zero."""
    zeros = _t([[0.0], [0.0]])
    loss = _call(zeros, zeros, zeros, _t([1.0, -1.0]), torch.tensor([[True], [True]]), kl_coef=0.0)
    assert torch.allclose(loss, _t(0.0), atol=1e-9), (
        "two-response average is wrong: average the per-response losses over the group"
    )
