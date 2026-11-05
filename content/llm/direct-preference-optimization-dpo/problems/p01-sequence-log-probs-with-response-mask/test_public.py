# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import torch
from submission import sequence_log_probs


def test_uniform_logits_sum_two_scored_tokens():
    """Uniform logits give each token probability 1/4; two response tokens sum to 2 log(1/4)."""
    logits = torch.zeros((1, 3, 4), dtype=torch.float64)
    input_ids = torch.tensor([[0, 2, 1]])
    mask = torch.tensor([[False, True, True]])
    out = sequence_log_probs(logits, input_ids, mask)
    assert out.shape == (1,), "output must have shape (B,): one sum per row"
    assert torch.allclose(
        out, torch.tensor([2 * math.log(0.25)], dtype=torch.float64), atol=1e-9
    ), (
        "uniform-logit sum is wrong: each response token contributes log(1/4), "
        "and the mask selects tokens"
    )


def test_logits_shift_one_position_to_score_the_next_token():
    """Token 1 is scored by the logits at position 0, which give probabilities [0.25, 0.75]."""
    logits = torch.tensor([[[0.0, math.log(3.0)], [0.0, 0.0]]], dtype=torch.float64)
    input_ids = torch.tensor([[0, 1]])
    mask = torch.tensor([[False, True]])
    out = sequence_log_probs(logits, input_ids, mask)
    assert torch.allclose(out, torch.tensor([math.log(0.75)], dtype=torch.float64), atol=1e-9), (
        "the one-token shift is wrong: the token at position t is scored by the "
        "logits at position t - 1"
    )


def test_row_without_response_tokens_sums_to_zero():
    """A row whose mask is all False has no scored tokens, so its sum is exactly zero."""
    logits = torch.zeros((1, 2, 3), dtype=torch.float64)
    input_ids = torch.tensor([[1, 2]])
    mask = torch.tensor([[False, False]])
    out = sequence_log_probs(logits, input_ids, mask)
    assert out.item() == 0.0, (
        "a row with no response tokens must sum to zero, not to a log-probability"
    )
