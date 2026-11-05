# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import gpt_forward

F64 = torch.float64


def _base_params(vocab, d, ctx):
    """Embeddings zero, no blocks, final LayerNorm gamma 1 and beta 0. The caller adds the rest."""
    return {
        "tok_emb": torch.zeros(vocab, d, dtype=F64),
        "pos_emb": torch.zeros(ctx, d, dtype=F64),
        "blocks": [],
        "ln_f_gamma": torch.ones(d, dtype=F64),
        "ln_f_beta": torch.zeros(d, dtype=F64),
    }


def _forward(idx, params, num_heads):
    """Call the learner; a crash on valid input is reported with the likely cause."""
    try:
        return gpt_forward(idx, params, num_heads)
    except Exception as exc:
        pytest.fail(
            f"valid input crashed ({type(exc).__name__}): check the parameter keys, the shapes, "
            "and the head (tied: tok_emb.T with shape (D, V))"
        )


def _zero_block(d, ff):
    """A block whose sub-layers add nothing: every projection and bias is zero, gammas are 1."""
    zeros = lambda *shape: torch.zeros(*shape, dtype=F64)  # noqa: E731
    ones = torch.ones(d, dtype=F64)
    return {
        "ln1_gamma": ones.clone(), "ln1_beta": zeros(d),
        "W_q": zeros(d, d), "b_q": zeros(d), "W_k": zeros(d, d), "b_k": zeros(d),
        "W_v": zeros(d, d), "b_v": zeros(d), "W_o": zeros(d, d), "b_o": zeros(d),
        "ln2_gamma": ones.clone(), "ln2_beta": zeros(d),
        "W_1": zeros(d, ff), "b_1": zeros(ff), "W_2": zeros(ff, d), "b_2": zeros(d),
    }  # fmt: skip


def test_tied_head_worked_example_gives_expected_logits():
    params = _base_params(vocab=3, d=2, ctx=1)
    params["tok_emb"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=F64)
    idx = torch.tensor([[1]])
    out = _forward(idx, params, num_heads=1)
    assert out.shape == (1, 1, 3), "logits must have shape (B, T, V)"
    expected = torch.tensor([[[-1.0, 1.0, 0.0]]], dtype=F64)
    assert torch.allclose(out, expected, atol=1e-4), (
        "tied logits differ: check the LayerNorm, and that the head is tok_emb transposed"
    )


def test_zero_block_passes_the_stream_through_to_the_same_logits():
    params = _base_params(vocab=3, d=2, ctx=1)
    params["tok_emb"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=F64)
    idx = torch.tensor([[1]])
    without = _forward(idx, params, num_heads=1)
    params["blocks"] = [_zero_block(2, 3)]
    with_block = _forward(idx, params, num_heads=1)
    assert torch.allclose(with_block, without, atol=1e-10), (
        "a block with zero sub-layers must leave the logits unchanged: check the residual adds"
    )


def test_token_id_outside_vocabulary_raises_value_error():
    params = _base_params(vocab=3, d=2, ctx=1)
    with pytest.raises(ValueError):
        gpt_forward(torch.tensor([[3]]), params, num_heads=1)
