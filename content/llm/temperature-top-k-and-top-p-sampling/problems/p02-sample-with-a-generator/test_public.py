# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import sample_token


def test_one_hot_distribution_always_returns_its_only_token():
    probs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    draws = [sample_token(probs, gen) for _ in range(20)]
    assert draws == [2] * 20, "a one-hot distribution must always return its only non-zero index"


def test_zero_probability_token_is_never_returned():
    probs = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    draws = {sample_token(probs, gen) for _ in range(200)}
    assert 1 not in draws, "a zero-probability token was returned: check the strict > against u"


def test_single_entry_vocabulary_returns_zero():
    gen = torch.Generator().manual_seed(1)
    assert sample_token(torch.tensor([1.0], dtype=torch.float64), gen) == 0, (
        "a one-token vocabulary must return index 0"
    )
