# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import top_k_route


def test_top_two_of_three_experts_are_gated_by_renormalised_softmax():
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    idx, gates, counts = top_k_route(logits, 2)
    assert idx.tolist() == [[0, 1]], "wrong experts: rank the logits and keep the k largest"
    torch.testing.assert_close(
        gates,
        torch.tensor([[0.7310585786300049, 0.2689414213699951]], dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="gates must be the softmax of the chosen logits (renormalised over the k chosen)",
    )
    assert counts.tolist() == [1, 1, 0], "expert_counts must count each chosen expert once"


def test_tied_logits_pick_lowest_expert_indices():
    logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    idx, gates, _ = top_k_route(logits, 2)
    assert idx.tolist() == [[0, 1]], "ties must go to the lower expert index"
    torch.testing.assert_close(
        gates,
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="equal logits must get equal gates",
    )


def test_zero_tokens_returns_empty_outputs_with_zero_counts():
    logits = torch.zeros(0, 4, dtype=torch.float64)
    idx, gates, counts = top_k_route(logits, 2)
    assert tuple(idx.shape) == (0, 2), "expert_indices must be (T, k) even when T is 0"
    assert tuple(gates.shape) == (0, 2), "gate_weights must be (T, k) even when T is 0"
    assert counts.tolist() == [0, 0, 0, 0], (
        "expert_counts must be all zero when there are no tokens"
    )
