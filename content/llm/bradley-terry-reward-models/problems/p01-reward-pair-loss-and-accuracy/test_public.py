# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import torch
from submission import bradley_terry_loss_and_accuracy


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_two_pairs_give_mean_loss_and_half_accuracy():
    """Two pairs with margins +2 and -1: the mean loss is 0.7200948 and one pair is correct."""
    loss, acc = bradley_terry_loss_and_accuracy(_t([2.0, 0.0]), _t([0.0, 1.0]))
    assert torch.allclose(loss, _t(0.7200948492805976), atol=1e-9, rtol=1e-9), (
        "loss is wrong: use -log(sigmoid(r_c - r_r)) per pair, then average over pairs"
    )
    assert torch.allclose(acc, _t(0.5), atol=1e-12), (
        "accuracy is wrong: count pairs with r_c > r_r and divide by the number of pairs"
    )


def test_tied_rewards_give_log_two_and_count_as_incorrect():
    """Equal rewards give margin 0: the loss is log 2 and a tie is not a correct pair."""
    loss, acc = bradley_terry_loss_and_accuracy(_t([1.0, 1.0]), _t([1.0, 1.0]))
    assert torch.allclose(loss, _t(math.log(2.0)), atol=1e-9, rtol=1e-9), (
        "tied pairs must give log 2 per pair: a margin of zero sits at the sigmoid midpoint"
    )
    assert acc.item() == 0.0, "ties were counted as correct: accuracy needs a strict r_c > r_r"


def test_empty_batch_is_rejected_with_value_error():
    """A batch with no pairs has no mean, so the function must raise ValueError."""
    try:
        bradley_terry_loss_and_accuracy(_t([]), _t([]))
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"empty batch raised {type(exc).__name__}: an empty batch must raise ValueError"
        ) from None
    raise AssertionError("empty batch was accepted: a batch with no pairs must raise ValueError")
