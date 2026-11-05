# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from submission import multi_head_self_attention as _learner


def _call(*args, **kwargs):
    """Call the learner; any exception on a valid input is reported as a named contract failure."""
    try:
        return _learner(*args, **kwargs)
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _invalid(*args, **kwargs):
    try:
        _learner(*args, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid argument was accepted instead of raising ValueError")


def _weights(gen, d, scale=0.5):
    def mat():
        return torch.randn(d, d, generator=gen, dtype=torch.float64) * scale

    def vec():
        return torch.randn(d, generator=gen, dtype=torch.float64) * scale

    return {
        "W_q": mat(),
        "W_k": mat(),
        "W_v": mat(),
        "W_o": mat(),
        "b_q": vec(),
        "b_k": vec(),
        "b_v": vec(),
        "b_o": vec(),
    }


def _run(x, w, heads, mask=None):
    return _call(
        x,
        w["W_q"],
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        w["b_o"],
        heads,
        mask=mask,
    )


def _oracle(x, w, heads, mask=None):
    """PyTorch MultiheadAttention loaded with the same weights (its mask is True = blocked)."""
    d = x.shape[-1]
    mha = torch.nn.MultiheadAttention(d, heads, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([w["W_q"], w["W_k"], w["W_v"]], dim=0))
        mha.in_proj_bias.copy_(torch.cat([w["b_q"], w["b_k"], w["b_v"]]))
        mha.out_proj.weight.copy_(w["W_o"])
        mha.out_proj.bias.copy_(w["b_o"])
    mha.eval()
    attn_mask = None if mask is None else ~mask
    out, _ = mha(x, x, x, attn_mask=attn_mask, need_weights=False)
    return out


def test_swapping_head_blocks_consistently_leaves_output_unchanged():
    # Heads are interchangeable: swap the first two head blocks of W_q, W_k, W_v (rows and biases)
    # and the matching input columns of W_o. A correct layer returns the same output.
    gen = torch.Generator().manual_seed(40)
    d, heads = 8, 2
    d_k = d // heads
    x = torch.randn(2, 4, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d)
    swap = torch.cat([torch.arange(d_k, 2 * d_k), torch.arange(0, d_k), torch.arange(2 * d_k, d)])
    swapped = {
        "W_q": w["W_q"][swap],
        "W_k": w["W_k"][swap],
        "W_v": w["W_v"][swap],
        "b_q": w["b_q"][swap],
        "b_k": w["b_k"][swap],
        "b_v": w["b_v"][swap],
        "W_o": w["W_o"][:, swap],
        "b_o": w["b_o"],
    }
    torch.testing.assert_close(
        _run(x, swapped, heads),
        _run(x, w, heads),
        atol=1e-9,
        rtol=1e-9,
        msg="heads are not interchangeable: the head split or merge mixes feature blocks",
    )


def test_each_head_reads_its_own_contiguous_feature_block():
    gen = torch.Generator().manual_seed(23)
    d, heads, length = 6, 3, 4
    d_k = d // heads
    x = torch.randn(1, length, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d)
    q = x @ w["W_q"].T + w["b_q"]
    k = x @ w["W_k"].T + w["b_k"]
    v = x @ w["W_v"].T + w["b_v"]
    pieces = []
    for j in range(heads):
        sl = slice(j * d_k, (j + 1) * d_k)
        s = q[..., sl] @ k[..., sl].transpose(-2, -1) / math.sqrt(d_k)
        pieces.append(torch.softmax(s, dim=-1) @ v[..., sl])
    merged = torch.cat(pieces, dim=-1)
    want = merged @ w["W_o"].T + w["b_o"]
    torch.testing.assert_close(
        _run(x, w, heads),
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="per-head output differs from the head-by-head reference: check the split and merge",
    )


def test_projections_use_nn_linear_layout_on_non_symmetric_weights():
    gen = torch.Generator().manual_seed(25)
    d = 4
    x = torch.randn(1, 3, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d)
    torch.testing.assert_close(
        _run(x, w, 1),
        _oracle(x, w, 1),
        atol=1e-9,
        rtol=1e-9,
        msg="weights are applied in the wrong orientation: a projection is x @ W.T + b",
    )


def test_scores_scale_by_per_head_width_not_model_width():
    gen = torch.Generator().manual_seed(24)
    d, heads = 16, 4  # d_k = 4, while sqrt(d_model) = 4 and sqrt(d_k) = 2 differ
    x = torch.randn(2, 6, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d, scale=1.0)
    torch.testing.assert_close(
        _run(x, w, heads),
        _oracle(x, w, heads),
        atol=1e-9,
        rtol=1e-9,
        msg="score scale is wrong: divide by sqrt(d_k) with d_k = d_model / num_heads",
    )


def test_matches_pytorch_multihead_attention_on_seeded_batches():
    gen = torch.Generator().manual_seed(21)
    for batch, length, heads in [(1, 4, 1), (2, 5, 2), (3, 6, 4)]:
        d = 8
        x = torch.randn(batch, length, d, generator=gen, dtype=torch.float64)
        w = _weights(gen, d)
        torch.testing.assert_close(
            _run(x, w, heads),
            _oracle(x, w, heads),
            atol=1e-9,
            rtol=1e-9,
            msg="output disagrees with the reference: check projections and head merge",
        )


def test_matches_reference_with_mask_on_seeded_batches():
    gen = torch.Generator().manual_seed(22)
    d, length = 8, 5
    x = torch.randn(2, length, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d)
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
    torch.testing.assert_close(
        _run(x, w, 2, mask=mask),
        _oracle(x, w, 2, mask=mask),
        atol=1e-9,
        rtol=1e-9,
        msg="masked output disagrees with the reference: check which mask entries are allowed",
    )


def test_diagonal_mask_keeps_only_self_attention():
    gen = torch.Generator().manual_seed(26)
    d, length = 4, 3
    x = torch.randn(1, length, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d)
    v = x @ w["W_v"].T + w["b_v"]
    want = v @ w["W_o"].T + w["b_o"]
    out = _run(x, w, 2, mask=torch.eye(length, dtype=torch.bool))
    torch.testing.assert_close(
        out,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="with only self allowed, each token must return its own value: check mask direction",
    )


def test_fully_masked_row_raises_value_error():
    gen = torch.Generator().manual_seed(27)
    x = torch.randn(1, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1] = False
    _invalid(
        x,
        w["W_q"],
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        w["b_o"],
        2,
        mask=mask,
    )


def test_output_bias_is_added_once_to_every_token():
    gen = torch.Generator().manual_seed(28)
    d, length = 4, 3
    x = torch.zeros(1, length, d, dtype=torch.float64)
    w = _weights(gen, d)
    # Zero inputs give every token the same keys, so each head averages the values: context = b_v.
    want = w["b_v"] @ w["W_o"].T + w["b_o"]
    out = _run(x, w, 2)
    for t in range(length):
        torch.testing.assert_close(
            out[0, t],
            want,
            atol=1e-9,
            rtol=1e-9,
            msg="output bias must be added exactly once per token: check the value bias path",
        )


def test_batch_rows_are_independent():
    gen = torch.Generator().manual_seed(29)
    d = 8
    x = torch.randn(3, 4, d, generator=gen, dtype=torch.float64)
    w = _weights(gen, d)
    together = _run(x, w, 2)
    separately = torch.cat([_run(x[b : b + 1], w, 2) for b in range(3)])
    torch.testing.assert_close(
        together,
        separately,
        atol=1e-12,
        rtol=0,
        msg="batch sequences must not influence each other: check the batch axis",
    )


def test_large_input_magnitudes_match_reference():
    gen = torch.Generator().manual_seed(30)
    d = 8
    x = torch.randn(2, 5, d, generator=gen, dtype=torch.float64) * 50.0
    w = _weights(gen, d)
    torch.testing.assert_close(
        _run(x, w, 2),
        _oracle(x, w, 2),
        atol=1e-9,
        rtol=1e-9,
        msg="large inputs: the softmax must stay finite, so use a stable softmax",
    )


def test_tiny_input_magnitudes_stay_finite_and_match_reference():
    gen = torch.Generator().manual_seed(31)
    d = 8
    x = torch.randn(2, 5, d, generator=gen, dtype=torch.float64) * 1e-12
    w = _weights(gen, d)
    out = _run(x, w, 2)
    assert torch.isfinite(out).all(), "tiny inputs must not produce NaN or infinity"
    torch.testing.assert_close(
        out,
        _oracle(x, w, 2),
        atol=1e-20,
        rtol=1e-7,
        msg="tiny inputs must follow the same formula: check scale and bias handling",
    )


def test_output_dtype_and_shape_follow_input():
    gen = torch.Generator().manual_seed(32)
    x = torch.randn(2, 3, 4, generator=gen, dtype=torch.float64)
    out = _run(x, _weights(gen, 4), 2)
    assert out.dtype == torch.float64, "output dtype must be float64 for float64 input"
    assert tuple(out.shape) == (2, 3, 4), "output shape must be (B, L, d_model)"


def test_three_dimensional_input_is_required():
    gen = torch.Generator().manual_seed(33)
    w = _weights(gen, 4)
    _invalid(
        torch.randn(3, 4, dtype=torch.float64),
        w["W_q"],
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        w["b_o"],
        2,
    )


def test_non_square_weight_raises_value_error():
    gen = torch.Generator().manual_seed(34)
    x = torch.randn(1, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    _invalid(
        x,
        torch.randn(4, 3, dtype=torch.float64),
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        w["b_o"],
        2,
    )


def test_bias_with_wrong_length_raises_value_error():
    gen = torch.Generator().manual_seed(35)
    x = torch.randn(1, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    _invalid(
        x,
        w["W_q"],
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        torch.zeros(1, 4, dtype=torch.float64),
        2,
    )


@pytest.mark.parametrize("heads", [0, -1], ids=["zero-heads", "negative-heads"])
def test_non_positive_head_count_raises_value_error(heads):
    gen = torch.Generator().manual_seed(36)
    x = torch.randn(1, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    _invalid(
        x, w["W_q"], w["W_k"], w["W_v"], w["W_o"], w["b_q"], w["b_k"], w["b_v"], w["b_o"], heads
    )


def test_mask_must_be_boolean():
    gen = torch.Generator().manual_seed(37)
    x = torch.randn(1, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    _invalid(
        x,
        w["W_q"],
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        w["b_o"],
        2,
        mask=torch.ones(3, 3, dtype=torch.float64),
    )


def test_mask_shape_must_match_sequence_length():
    gen = torch.Generator().manual_seed(38)
    x = torch.randn(1, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    _invalid(
        x,
        w["W_q"],
        w["W_k"],
        w["W_v"],
        w["W_o"],
        w["b_q"],
        w["b_k"],
        w["b_v"],
        w["b_o"],
        2,
        mask=torch.ones(2, 2, dtype=torch.bool),
    )


def test_inputs_are_not_mutated():
    gen = torch.Generator().manual_seed(39)
    x = torch.randn(2, 3, 4, generator=gen, dtype=torch.float64)
    w = _weights(gen, 4)
    before_x = x.clone()
    before_wq = w["W_q"].clone()
    _run(x, w, 2)
    assert torch.equal(x, before_x), "the input X must not be modified"
    assert torch.equal(w["W_q"], before_wq), "weights must not be modified in place"
