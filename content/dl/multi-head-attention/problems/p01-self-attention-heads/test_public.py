# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import multi_head_self_attention


def _identity_case():
    eye = torch.eye(2, dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64)
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    return x, eye, zero


def test_identity_weights_give_softmax_of_scaled_dot_products():
    x, eye, zero = _identity_case()
    out = multi_head_self_attention(x, eye, eye, eye, eye, zero, zero, zero, zero, num_heads=1)
    p = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1.0)
    want = torch.tensor([p, 1 - p], dtype=torch.float64)
    torch.testing.assert_close(
        out[0, 0],
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="row 0 must weight the values by softmax(q.k / sqrt(d_k)): check the scale",
    )


def test_mask_true_means_allowed_so_masked_row_copies_its_own_value():
    x, eye, zero = _identity_case()
    mask = torch.tensor([[True, False], [True, True]])
    out = multi_head_self_attention(
        x, eye, eye, eye, eye, zero, zero, zero, zero, num_heads=1, mask=mask
    )
    torch.testing.assert_close(
        out[0, 0],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        atol=1e-9,
        rtol=0,
        msg="a False entry must get zero weight: check that True means allowed",
    )


def test_num_heads_that_does_not_divide_d_model_raises_value_error():
    x, eye, zero = _identity_case()
    with pytest.raises(ValueError):
        multi_head_self_attention(x, eye, eye, eye, eye, zero, zero, zero, zero, num_heads=3)
