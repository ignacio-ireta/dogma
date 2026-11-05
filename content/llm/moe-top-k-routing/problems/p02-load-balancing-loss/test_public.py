# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import load_balance_loss


def test_balanced_router_scores_exactly_alpha():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    loss = load_balance_loss(logits)
    assert isinstance(loss, torch.Tensor), "return a torch tensor, not a Python float"
    torch.testing.assert_close(
        loss,
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="balanced routing must score alpha: check the factor E and the mean over tokens",
    )


def test_three_token_worked_example_matches_the_hand_computation():
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    loss = load_balance_loss(logits)
    assert isinstance(loss, torch.Tensor), "return a torch tensor, not a Python float"
    assert abs(loss.item() - 1.0846215728839737) < 1e-9, (
        "loss must be alpha * E * sum_i f_i * P_i with f from top-1 counts and P from softmax"
    )


def test_zero_tokens_raise_value_error():
    with pytest.raises(ValueError):
        load_balance_loss(torch.zeros(0, 3, dtype=torch.float64))
