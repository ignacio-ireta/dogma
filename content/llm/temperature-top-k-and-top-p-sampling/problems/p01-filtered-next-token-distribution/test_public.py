# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import filtered_distribution


def test_uniform_logits_give_uniform_distribution():
    out = filtered_distribution(torch.zeros(4, dtype=torch.float64), temperature=1.0)
    torch.testing.assert_close(
        out,
        torch.full((4,), 0.25, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="uniform logits must give a uniform distribution: check the softmax normalisation",
    )


def test_top_p_keeps_the_smallest_prefix_that_reaches_p():
    logits = torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64))
    out = filtered_distribution(logits, temperature=1.0, top_p=0.7)
    want = torch.tensor([2 / 3, 1 / 3, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        out,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="top-p must keep the smallest prefix reaching p and renormalise the kept mass",
    )


def test_top_k_tie_at_the_boundary_keeps_the_lower_index():
    logits = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    out = filtered_distribution(logits, temperature=1.0, top_k=1)
    want = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        out,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="top-k tie must keep exactly k tokens, the lower index first",
    )


def test_zero_temperature_is_rejected_with_value_error():
    try:
        filtered_distribution(torch.zeros(2, dtype=torch.float64), temperature=0.0)
    except ValueError:
        return
    raise AssertionError("temperature 0 must be rejected: greedy decoding is a separate rule")
