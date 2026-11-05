# Public tests: the examples from problem.md plus one edge case.
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import causal_padding_mask


def test_square_mask_is_lower_triangle():
    """L == S gives the ordinary lower triangle."""
    out = causal_padding_mask(2, 2)
    expected = torch.tensor([[True, False], [True, True]])
    assert out.dtype == torch.bool, "mask must be boolean"
    assert torch.equal(out, expected), "square mask must be the lower triangle (j <= i)"


def test_decode_step_is_bottom_right_aligned():
    """Two new queries after a cache of two: the offset is S - L = 2."""
    out = causal_padding_mask(2, 4)
    expected = torch.tensor([[True, True, True, False], [True, True, True, True]])
    assert torch.equal(out, expected), (
        "wrong alignment: a KV-cache step must align queries to the last keys, not the first"
    )


def test_fully_masked_row_is_all_false_and_no_error():
    """Edge case: with L > S, query 0 has no allowed key. Its row is all False, without an error."""
    out = causal_padding_mask(3, 2)
    expected = torch.tensor([[False, False], [True, False], [True, True]])
    assert torch.equal(out, expected), (
        "fully masked rows must be all False; never set them to True or raise"
    )
