# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import greedy_generate

EOS = 3
PAD = 0


def _next_mod4_model(tokens: torch.Tensor) -> torch.Tensor:
    """Toy model: the argmax of the logits is (last token + 1) % 4."""
    eye = torch.eye(4, dtype=torch.float64)
    return eye[(tokens[:, -1] + 1) % 4]


def test_each_row_stops_at_its_own_eos_and_is_padded():
    prompts = torch.tensor([[0], [1]])
    out = greedy_generate(_next_mod4_model, prompts, 5, eos_id=EOS, pad_id=PAD)
    expected = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    assert torch.equal(out, expected), "rows must stop at their own EOS and pad afterwards"


def test_step_cap_stops_the_loop_before_eos():
    prompts = torch.tensor([[0], [1]])
    out = greedy_generate(_next_mod4_model, prompts, 2, eos_id=EOS, pad_id=PAD)
    expected = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert torch.equal(out, expected), "max_new_tokens must cap the number of new tokens"


def test_zero_new_tokens_returns_the_prompts_unchanged():
    prompts = torch.tensor([[2, 1], [0, 0]])
    out = greedy_generate(_next_mod4_model, prompts, 0, eos_id=EOS, pad_id=PAD)
    assert out.shape == prompts.shape, "a zero step cap must return the prompt width unchanged"
    assert torch.equal(out, prompts), "a zero step cap must return the prompts, not new tokens"
