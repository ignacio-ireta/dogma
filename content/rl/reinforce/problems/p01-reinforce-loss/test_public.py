from __future__ import annotations

import math

import torch
from submission import reinforce_loss


def test_worked_example_matches_minus_two_log_half():
    logits = torch.zeros(1, 2, 2, dtype=torch.float64)
    actions = torch.tensor([[1, 0]])
    rewards = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    loss = reinforce_loss(logits, actions, rewards, gamma=1.0)
    assert math.isclose(float(loss), -2 * math.log(0.5), rel_tol=1e-9), (
        "loss must be -(1/B) sum log pi * G with G = [1, 1] and log pi = log 0.5"
    )


def test_gamma_zero_with_baseline_gives_zero_for_one_episode():
    logits = torch.zeros(1, 2, 2, dtype=torch.float64)
    actions = torch.tensor([[1, 0]])
    rewards = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    loss = reinforce_loss(logits, actions, rewards, gamma=0.0, baseline=True)
    assert abs(float(loss)) < 1e-12, (
        "one episode with a batch-mean baseline has zero advantage, so the loss is zero"
    )


def test_invalid_gamma_is_rejected():
    logits = torch.zeros(1, 1, 2)
    try:
        reinforce_loss(logits, torch.tensor([[0]]), torch.tensor([[1.0]]), gamma=1.5)
    except ValueError:
        return
    raise AssertionError("gamma outside [0, 1] must raise ValueError")
