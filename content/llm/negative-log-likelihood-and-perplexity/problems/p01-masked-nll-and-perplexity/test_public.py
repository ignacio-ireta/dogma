# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import torch
from submission import masked_nll_and_perplexity


def test_uniform_logits_give_log_vocab_and_perplexity_equal_to_vocab():
    """Uniform guess over V = 4: NLL is log 4 and perplexity is 4."""
    logits = torch.zeros((1, 2, 4), dtype=torch.float64)
    targets = torch.tensor([[0, 3]])
    mean_nll, ppl = masked_nll_and_perplexity(logits, targets)
    assert abs(float(mean_nll) - math.log(4)) < 1e-9, "uniform logits must give NLL log V in nats"
    assert abs(float(ppl) - 4.0) < 1e-9, "perplexity of a uniform guess over V must equal V"


def test_mean_is_over_scored_tokens_and_ignored_positions_are_skipped():
    """Edge case: the ignored middle position does not count in the denominator."""
    logits = torch.zeros((1, 3, 4), dtype=torch.float64)
    targets = torch.tensor([[1, -100, 2]])
    mean_nll, _ = masked_nll_and_perplexity(logits, targets)
    assert abs(float(mean_nll) - math.log(4)) < 1e-9, (
        "the mean must divide by the 2 scored tokens, not by all 3 positions"
    )


def test_peaked_example_matches_hand_computed_mean_and_perplexity():
    logits = torch.tensor([[[2.0, 0.0]], [[0.0, 0.0]]], dtype=torch.float64)
    targets = torch.tensor([[0], [1]])
    mean_nll, ppl = masked_nll_and_perplexity(logits, targets)
    want = (math.log(1 + math.exp(-2)) + math.log(2)) / 2
    assert abs(float(mean_nll) - want) < 1e-9, (
        "mean NLL must be the average of the two per-token NLLs"
    )
    assert abs(float(ppl) - math.exp(want)) < 1e-9, "perplexity must be exp of the mean NLL"
