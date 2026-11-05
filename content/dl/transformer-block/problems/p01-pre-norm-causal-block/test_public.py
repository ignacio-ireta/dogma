# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import pre_norm_block


def _zero_params(d, ff):
    """Every projection and bias is zero; both LayerNorms have gamma 1 and beta 0."""
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64)  # noqa: E731
    ones = torch.ones(d, dtype=torch.float64)
    return {
        "ln1_gamma": ones.clone(), "ln1_beta": zeros(d),
        "W_q": zeros(d, d), "b_q": zeros(d), "W_k": zeros(d, d), "b_k": zeros(d),
        "W_v": zeros(d, d), "b_v": zeros(d), "W_o": zeros(d, d), "b_o": zeros(d),
        "ln2_gamma": ones.clone(), "ln2_beta": zeros(d),
        "W_1": zeros(d, ff), "b_1": zeros(ff), "W_2": zeros(ff, d), "b_2": zeros(d),
    }  # fmt: skip


def test_zero_projections_make_the_block_the_identity():
    x = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)
    out = pre_norm_block(x, _zero_params(2, 3), num_heads=1)
    assert torch.allclose(out, x), "with zero sub-layers the block must pass x through unchanged"


def test_two_token_causal_example_matches_worked_values():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    p = _zero_params(2, 2)
    eye = torch.eye(2, dtype=torch.float64)
    for key in ("W_q", "W_k", "W_v", "W_o", "W_1"):
        p[key] = eye.clone()
    out = pre_norm_block(x, p, num_heads=1)
    expected = torch.tensor(
        [[[2.0, -1.0], [-0.8884, 1.8884]]], dtype=torch.float64
    )  # token 1 weights token 0 by about 0.056, itself by about 0.944
    assert torch.allclose(out, expected, atol=1e-4), (
        "causal example differs: check the mask and the scaling of the attention scores"
    )


def test_heads_must_divide_the_model_width():
    x = torch.ones(1, 1, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        pre_norm_block(x, _zero_params(2, 3), num_heads=3)
