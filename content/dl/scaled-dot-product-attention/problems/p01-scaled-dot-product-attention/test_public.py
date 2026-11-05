# Public tests: the examples from problem.md plus one edge case.
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import scaled_dot_product_attention

DT = torch.float64


def _example():
    q = torch.tensor([[1.0, 0.0]], dtype=DT)
    k = torch.eye(2, dtype=DT)
    v = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=DT)
    return q, k, v


def test_unmasked_example_weights_and_output():
    """Scores [1, 0] / sqrt(2) give weights [0.6698, 0.3302] and output [6.6976, 3.3024]."""
    out, w = scaled_dot_product_attention(*_example())
    assert torch.allclose(w, torch.tensor([[0.6697615, 0.3302385]], dtype=DT), atol=1e-6), (
        "weights differ from the worked example: check the 1/sqrt(d_k) scale "
        "and the key-axis softmax"
    )
    assert torch.allclose(out, torch.tensor([[6.697615, 3.302385]], dtype=DT), atol=1e-5), (
        "output differs from the worked example: the output must be weights @ v"
    )


def test_boolean_mask_removes_second_key():
    """A mask that allows only key 0 gives weights [1, 0] and output [10, 0]."""
    q, k, v = _example()
    out, w = scaled_dot_product_attention(q, k, v, torch.tensor([[True, False]]))
    assert torch.allclose(w, torch.tensor([[1.0, 0.0]], dtype=DT)), (
        "masked key still receives weight: True must mean may attend"
    )
    assert torch.allclose(out, torch.tensor([[10.0, 0.0]], dtype=DT)), (
        "output used a masked value: weights must be zero where the mask is False"
    )


def test_fully_masked_row_is_zero_not_nan():
    """Edge case: a query with no allowed key has zero weights and zero output."""
    q, k, v = _example()
    out, w = scaled_dot_product_attention(q, k, v, torch.tensor([[False, False]]))
    assert torch.isfinite(out).all() and torch.isfinite(w).all(), (
        "fully masked row produced NaN or inf: the naive softmax over all -inf is the trap"
    )
    assert torch.count_nonzero(w) == 0 and torch.count_nonzero(out) == 0, (
        "fully masked row must have zero weights and zero output"
    )
