# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show
# hidden inputs, shapes or expected values. The oracle uses torch.nn.functional pieces, which the
# learner may not import.
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias for the oracle
from submission import gpt_forward

EPS = 1e-5
ATOL = 1e-10
RTOL = 1e-10
BLOCK_KEYS = (
    "ln1_gamma", "ln1_beta", "W_q", "b_q", "W_k", "b_k", "W_v", "b_v",
    "W_o", "b_o", "ln2_gamma", "ln2_beta", "W_1", "b_1", "W_2", "b_2",
)  # fmt: skip


def _random_block(d, ff, gen):
    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    return {
        "ln1_gamma": 1 + 0.2 * rnd(d), "ln1_beta": 0.1 * rnd(d),
        "W_q": rnd(d, d) / math.sqrt(d), "b_q": 0.1 * rnd(d),
        "W_k": rnd(d, d) / math.sqrt(d), "b_k": 0.1 * rnd(d),
        "W_v": rnd(d, d) / math.sqrt(d), "b_v": 0.1 * rnd(d),
        "W_o": rnd(d, d) / math.sqrt(d), "b_o": 0.1 * rnd(d),
        "ln2_gamma": 1 + 0.2 * rnd(d), "ln2_beta": 0.1 * rnd(d),
        "W_1": rnd(d, ff) / math.sqrt(d), "b_1": 0.1 * rnd(ff),
        "W_2": rnd(ff, d) / math.sqrt(ff), "b_2": 0.1 * rnd(d),
    }  # fmt: skip


def _random_params(vocab, d, ctx, n_blocks, ff, gen, *, tied):  # noqa: PLR0917
    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    p = {
        "tok_emb": 0.5 * rnd(vocab, d),
        "pos_emb": 0.3 * rnd(ctx, d),
        "blocks": [_random_block(d, ff, gen) for _ in range(n_blocks)],
        "ln_f_gamma": 1 + 0.2 * rnd(d),
        "ln_f_beta": 0.1 * rnd(d),
    }
    if not tied:
        p["lm_head"] = rnd(d, vocab) / math.sqrt(d)
    return p


def _ref_block(x, p, heads, *, causal=True):
    """Pre-norm causal block from F.layer_norm, F.linear, F.gelu and SDPA."""
    batch, tokens, width = x.shape
    dh = width // heads
    h = F.layer_norm(x, (width,), p["ln1_gamma"], p["ln1_beta"], EPS)
    q = F.linear(h, p["W_q"].T, p["b_q"]).reshape(batch, tokens, heads, dh).transpose(1, 2)
    k = F.linear(h, p["W_k"].T, p["b_k"]).reshape(batch, tokens, heads, dh).transpose(1, 2)
    v = F.linear(h, p["W_v"].T, p["b_v"]).reshape(batch, tokens, heads, dh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    a = F.linear(o.transpose(1, 2).reshape(batch, tokens, width), p["W_o"].T, p["b_o"])
    x = x + a
    h2 = F.layer_norm(x, (width,), p["ln2_gamma"], p["ln2_beta"], EPS)
    f = F.linear(F.gelu(F.linear(h2, p["W_1"].T, p["b_1"])), p["W_2"].T, p["b_2"])
    return x + f


def _ref_gpt(idx, p, heads, *, final_ln=True, positions="correct", causal=True, head="learned"):
    """Reference logits. The tied head uses tok_emb.T; an untied head uses lm_head.

    The keyword flags switch off one step, so the failure message can name that mistake.
    """
    tokens = idx.shape[1]
    width = p["tok_emb"].shape[1]
    x = F.embedding(idx, p["tok_emb"])
    if positions == "correct":
        x = x + p["pos_emb"][:tokens]
    elif positions == "row0":  # the same position row for every token
        x = x + p["pos_emb"][0]
    for block in p["blocks"]:
        x = _ref_block(x, block, heads, causal=causal)
    h = F.layer_norm(x, (width,), p["ln_f_gamma"], p["ln_f_beta"], EPS) if final_ln else x
    if head == "tied" or "lm_head" not in p:
        weight = p["tok_emb"]
    else:
        weight = p["lm_head"].T
    return F.linear(h, weight)


def _run(idx, p, heads, *, pass_through=False):
    """Call the learner on copies, so an in-place write cannot corrupt the oracle's inputs."""
    if pass_through:
        call_idx, call_p = idx, p
    else:
        call_idx = idx.clone()
        call_p = {
            key: (
                [{k: v.clone() for k, v in b.items()} for b in value]
                if key == "blocks"
                else value.clone()
            )
            for key, value in p.items()
        }
    try:
        out = gpt_forward(call_idx, call_p, heads)
    except Exception as exc:  # any crash on valid input is a contract failure
        pytest.fail(
            f"valid input was rejected or crashed ({type(exc).__name__}): check the parameter "
            "keys, the shapes, and the head (tied: tok_emb.T with shape (D, V))"
        )
    if not torch.is_tensor(out):
        pytest.fail("return a tensor of logits, not a Python object")
    if tuple(out.shape) != (idx.shape[0], idx.shape[1], p["tok_emb"].shape[0]):
        pytest.fail("logits must have shape (B, T, V): check the head's output axis")
    if out.dtype != p["tok_emb"].dtype:
        pytest.fail("logits dtype must match the parameter dtype")
    if not torch.isfinite(out).all():
        pytest.fail("non-finite logits: check the attention mask and the LayerNorm epsilon")
    return out


MISTAKES = [
    (
        {"final_ln": False},
        "final LayerNorm missing: normalise the stream once more before the LM head",
    ),
    (
        {"positions": "none"},
        "position embeddings missing: add pos_emb[t] to each token embedding",
    ),
    (
        {"positions": "row0"},
        "position row misindexed: use row t of pos_emb for position t, not row 0",
    ),
    (
        {"causal": False},
        "causal mask missing: query i must not attend to keys j > i",
    ),
    (
        {"head": "tied"},
        "lm_head ignored: when params has lm_head, project onto lm_head, not tok_emb.T",
    ),
]


def _assert_matches(out, idx, p, heads):
    expected = _ref_gpt(idx, p, heads)
    if torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        return
    for variant, message in MISTAKES:
        if torch.allclose(out, _ref_gpt(idx, p, heads, **variant), atol=ATOL, rtol=RTOL):
            pytest.fail(message)
    pytest.fail(
        "logits differ from the GPT reference: check the embedding sum and position rows, "
        "the LayerNorm (biased variance, eps 1e-5), the attention scaling and head split, "
        "the GELU form, the bias terms, the block order, the final LayerNorm and the head"
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 5, 4, 4, 0, 1, 6, True),
        (2, 5, 7, 8, 8, 1, 2, 12, True),
        (3, 4, 6, 6, 6, 2, 3, 5, False),
        (2, 7, 11, 12, 10, 3, 4, 20, True),
        (1, 3, 4, 2, 5, 2, 2, 1, False),
    ],
    ids=["no-blocks-tied", "one-block-tied", "two-blocks-untied", "three-blocks-tied",
         "narrow-untied"],
)  # fmt: skip
def test_matches_gpt_reference_on_seeded_batches(shape):
    batch, tokens, vocab, width, ctx, n_blocks, heads, ff, tied = shape
    gen = torch.Generator().manual_seed(batch * 100 + tokens + n_blocks)
    p = _random_params(vocab, width, ctx, n_blocks, ff, gen, tied=tied)
    idx = torch.randint(0, vocab, (batch, tokens), generator=gen)
    _assert_matches(_run(idx, p, heads), idx, p, heads)


def test_tied_head_shares_the_token_table_with_an_explicit_transpose():
    gen = torch.Generator().manual_seed(29)
    p = _random_params(6, 4, 5, 1, 7, gen, tied=True)
    idx = torch.randint(0, 6, (2, 3), generator=gen)
    tied_out = _run(idx, p, 2)
    untied = dict(p)
    untied["lm_head"] = p["tok_emb"].T.clone()
    explicit_out = _run(idx, untied, 2)
    if not torch.allclose(tied_out, explicit_out, atol=ATOL, rtol=RTOL):
        pytest.fail(
            "tied head does not use tok_emb: the output must match the head set to tok_emb.T"
        )


def test_tied_head_logits_move_when_the_token_table_moves():
    gen = torch.Generator().manual_seed(31)
    p = _random_params(5, 4, 4, 0, 6, gen, tied=True)
    idx = torch.tensor([[0, 1, 2]])
    before = _run(idx, p, 2)
    p2 = dict(p)
    p2["tok_emb"] = p["tok_emb"] * 1.5
    after = _run(idx, p2, 2)
    expected = _ref_gpt(idx, p2, 2)
    if torch.allclose(before, after):
        pytest.fail("tied head ignores tok_emb: logits did not change when the table changed")
    if not torch.allclose(after, expected, atol=ATOL, rtol=RTOL):
        pytest.fail("tied head disagrees with the reference after the table changed")


def test_future_tokens_do_not_change_earlier_logits():
    gen = torch.Generator().manual_seed(37)
    p = _random_params(9, 6, 6, 2, 8, gen, tied=True)
    idx = torch.randint(0, 9, (2, 5), generator=gen)
    before = _run(idx, p, 2)
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 9
    after = _run(changed, p, 2)
    if not torch.allclose(before[:, :-1], after[:, :-1], atol=ATOL, rtol=RTOL):
        pytest.fail(
            "a later token changed earlier logits: the causal mask is missing or misaligned"
        )


def test_position_table_is_added_per_position_not_per_row():
    gen = torch.Generator().manual_seed(41)
    p = _random_params(5, 4, 4, 0, 6, gen, tied=True)
    idx = torch.tensor([[2, 2, 2], [2, 2, 2]])
    out = _run(idx, p, 2)
    if torch.allclose(out[0, 0], out[0, 1]):
        pytest.fail("identical tokens at different positions gave equal logits: add pos_emb[t]")
    if not torch.allclose(out, _ref_gpt(idx, p, 2), atol=ATOL, rtol=RTOL):
        pytest.fail("position embeddings are misindexed: use row t of pos_emb for position t")


def test_batch_rows_are_processed_independently():
    gen = torch.Generator().manual_seed(43)
    p = _random_params(7, 6, 5, 2, 8, gen, tied=False)
    idx = torch.randint(0, 7, (3, 4), generator=gen)
    together = _run(idx, p, 3)
    for row in range(idx.shape[0]):
        alone = _run(idx[row : row + 1], p, 3)
        if not torch.allclose(together[row : row + 1], alone, atol=ATOL, rtol=RTOL):
            pytest.fail("batch rows interact: a reshape or a mask mixes examples in the batch")


def test_context_length_boundary_is_accepted():
    gen = torch.Generator().manual_seed(47)
    p = _random_params(5, 4, 3, 1, 6, gen, tied=True)
    idx = torch.randint(0, 5, (1, 3), generator=gen)
    _assert_matches(_run(idx, p, 2), idx, p, 2)


def test_sequence_longer_than_context_raises_value_error():
    gen = torch.Generator().manual_seed(53)
    p = _random_params(5, 4, 3, 0, 6, gen, tied=True)
    idx = torch.zeros(1, 4, dtype=torch.int64)
    _expect_value_error(lambda: gpt_forward(idx, p, 2), "sequence longer than context was accepted")


@pytest.mark.parametrize("scale", [1e4, 1e-4], ids=["large-magnitude", "small-magnitude"])
def test_extreme_embedding_scales_match_the_reference(scale):
    gen = torch.Generator().manual_seed(59)
    p = _random_params(6, 4, 4, 1, 6, gen, tied=False)
    p["tok_emb"] = scale * p["tok_emb"]
    p["pos_emb"] = scale * p["pos_emb"]
    idx = torch.randint(0, 6, (2, 3), generator=gen)
    _assert_matches(_run(idx, p, 2), idx, p, 2)


def test_float32_parameters_return_float32_close_to_reference():
    gen = torch.Generator().manual_seed(61)
    p64 = _random_params(6, 8, 5, 1, 12, gen, tied=True)
    idx = torch.randint(0, 6, (2, 4), generator=gen)
    p32 = {
        key: (
            [{k: v.float() for k, v in b.items()} for b in value]
            if key == "blocks"
            else value.float()
        )
        for key, value in p64.items()
    }
    out = _run(idx, p32, 2)
    if out.dtype != torch.float32:
        pytest.fail("float32 parameters must return float32 logits")
    if not torch.allclose(out.double(), _ref_gpt(idx, p64, 2), atol=1e-4, rtol=1e-4):
        pytest.fail("float32 logits drift from the reference beyond single-precision round-off")


def test_inputs_and_parameters_are_not_mutated():
    gen = torch.Generator().manual_seed(67)
    p = _random_params(6, 4, 4, 1, 6, gen, tied=False)
    idx = torch.randint(0, 6, (2, 3), generator=gen)
    idx_before, p_before = idx.clone(), copy.deepcopy(p)
    _run(idx, p, 2, pass_through=True)
    if not torch.equal(idx, idx_before):
        pytest.fail("the token ids were modified in place: never write into idx")
    for key in p:
        if key == "blocks":
            for mine, before in zip(p[key], p_before[key], strict=True):
                for name in BLOCK_KEYS:
                    if not torch.equal(mine[name], before[name]):
                        pytest.fail(
                            "a block parameter was modified in place: never write into params"
                        )
        elif not torch.equal(p[key], p_before[key]):
            pytest.fail("a parameter tensor was modified in place: never write into params")


def _valid(seed=71, *, tied=True, n_blocks=1):
    gen = torch.Generator().manual_seed(seed)
    p = _random_params(5, 4, 4, n_blocks, 6, gen, tied=tied)
    idx = torch.tensor([[1, 2, 3]])
    return idx, p


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a value instead of raising ValueError")


def _set(key, value):
    def mutate(idx, p):
        p[key] = value
        return idx, p, 2

    return mutate


def _set_block(key, value):
    def mutate(idx, p):
        p["blocks"][0][key] = value
        return idx, p, 2

    return mutate


def _drop(key, mapping="top"):
    def mutate(idx, p):
        if mapping == "top":
            del p[key]
        else:
            p["blocks"][0] = {k: v for k, v in p["blocks"][0].items() if k != key}
        return idx, p, 2

    return mutate


def _ids(value):
    def mutate(idx, p):
        return value, p, 2

    return mutate


_F64_ZERO = torch.zeros(1, dtype=torch.float64)

INVALID_CASES = {
    "heads-do-not-divide": lambda idx, p: (idx, p, 3),
    "zero-heads": lambda idx, p: (idx, p, 0),
    "bool-heads": lambda idx, p: (idx, p, True),
    "float-ids": _ids(_valid()[0].float()),
    "two-dimensional-ids": _ids(torch.tensor([1, 2, 3])),
    "empty-ids": _ids(torch.zeros(1, 0, dtype=torch.int64)),
    "negative-id": _ids(torch.tensor([[1, -1, 2]])),
    "id-at-vocabulary-size": _ids(torch.tensor([[1, 5, 2]])),
    "missing-key": _drop("pos_emb"),
    "extra-key": _set("dropout", _F64_ZERO),
    "lm-head-wrong-shape": _set("lm_head", torch.zeros(4, 4, dtype=torch.float64)),
    "token-table-wrong-shape": _set("tok_emb", torch.zeros(5, 3, dtype=torch.float64)),
    "block-missing-key": _drop("W_o", mapping="block"),
    "block-wrong-shape": _set_block("W_q", torch.zeros(4, 3, dtype=torch.float64)),
    "param-dtype-mismatch": _set("ln_f_beta", _valid()[1]["ln_f_beta"].float()),
    "param-not-a-tensor": _set("ln_f_gamma", [1.0, 1.0, 1.0, 1.0]),
    "blocks-not-a-list": _set("blocks", {"W_q": 1}),
}


@pytest.mark.parametrize("case", list(INVALID_CASES), ids=list(INVALID_CASES))
def test_invalid_inputs_raise_value_error(case):
    idx, p = _valid()
    call_idx, call_p, heads = INVALID_CASES[case](idx, p)
    _expect_value_error(
        lambda: gpt_forward(call_idx, call_p, heads), f"invalid input ({case}) was accepted"
    )


def test_untied_head_in_another_dtype_raises_value_error():
    idx, p = _valid(tied=False)
    p["lm_head"] = p["lm_head"].float()
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "an untied head in a different dtype from tok_emb was accepted: every parameter must share "
        "the dtype of tok_emb",
    )


def test_position_table_in_another_dtype_raises_value_error():
    idx, p = _valid()
    p["pos_emb"] = p["pos_emb"].float()
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a position table in a different dtype from tok_emb was accepted: every parameter must "
        "share the dtype of tok_emb",
    )


def test_position_table_with_wrong_width_raises_value_error():
    idx, p = _valid()
    p["pos_emb"] = torch.zeros(4, 5, dtype=torch.float64)  # width must equal D
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a position table whose width differs from D was accepted: pos_emb must have shape (C, D)",
    )


def test_block_parameter_in_another_dtype_raises_value_error():
    idx, p = _valid()
    p["blocks"][0]["W_q"] = p["blocks"][0]["W_q"].float()
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a block parameter in a different dtype from tok_emb was accepted: every parameter must "
        "share the dtype of tok_emb",
    )


def test_block_parameter_that_is_not_a_tensor_raises_value_error():
    idx, p = _valid()
    p["blocks"][0]["W_q"] = p["blocks"][0]["W_q"].tolist()
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a block parameter given as a plain list was accepted or failed with the wrong error: "
        "every block parameter must be a tensor",
    )


def test_non_floating_token_table_raises_value_error():
    ints = {
        "tok_emb": torch.zeros(5, 4, dtype=torch.int64),
        "pos_emb": torch.zeros(4, 4, dtype=torch.int64),
        "blocks": [],
        "ln_f_gamma": torch.ones(4, dtype=torch.int64),
        "ln_f_beta": torch.zeros(4, dtype=torch.int64),
    }
    idx = torch.tensor([[1, 2, 3]])
    _expect_value_error(
        lambda: gpt_forward(idx, ints, 2),
        "an integer token table was accepted or failed with the wrong error: tok_emb must be "
        "floating-point and raise ValueError",
    )


def test_one_dimensional_feed_forward_weight_raises_value_error():
    idx, p = _valid()
    p["blocks"][0]["W_1"] = torch.zeros(4, dtype=torch.float64)
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a one-dimensional W_1 was accepted or failed with the wrong error: W_1 must be (D, F)",
    )


def test_zero_width_feed_forward_block_raises_value_error():
    idx, p = _valid()
    block = p["blocks"][0]
    block["W_1"] = torch.zeros(4, 0, dtype=torch.float64)
    block["b_1"] = torch.zeros(0, dtype=torch.float64)
    block["W_2"] = torch.zeros(0, 4, dtype=torch.float64)
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a block with zero feed-forward width was accepted: the width F must be at least 1",
    )


def test_params_that_is_not_a_mapping_raises_value_error():
    idx, _ = _valid()
    _expect_value_error(
        lambda: gpt_forward(idx, None, 2),
        "params that is not a mapping was accepted or failed with the wrong error: "
        "params must be a mapping",
    )


def test_blocks_that_is_not_a_list_raises_value_error():
    idx, p = _valid()
    p["blocks"] = None
    _expect_value_error(
        lambda: gpt_forward(idx, p, 2),
        "a blocks value that is not a list was accepted or failed with the wrong error: "
        "blocks must be a list of block mappings",
    )
