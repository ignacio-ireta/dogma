# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show
# hidden inputs, shapes or expected values. Oracles use torch.nn.functional pieces and
# torch.nn.TransformerEncoderLayer, which the learner may not import.
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias for the oracle
from submission import pre_norm_block

EPS = 1e-5
ATOL = 1e-10  # float64 round-off through two sub-layers is far below this
RTOL = 1e-10


def _random_params(d, ff, gen):
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


def _ln(u, gamma, beta, *, unbiased=False, eps=EPS, eps_outside=False):
    if unbiased or eps != EPS or eps_outside:
        d = u.shape[-1]
        mean = u.mean(-1, keepdim=True)
        ss = ((u - mean) ** 2).sum(-1, keepdim=True)
        var = ss / (d - 1) if unbiased else ss / d
        if eps_outside:
            return gamma * (u - mean) / (torch.sqrt(var) + eps) + beta
        return gamma * (u - mean) / torch.sqrt(var + eps) + beta
    return F.layer_norm(u, (u.shape[-1],), gamma, beta, EPS)


def _attn(h, p, heads, *, causal=True, head_layout="correct", width_scale=False):
    q = F.linear(h, p["W_q"].T, p["b_q"])
    k = F.linear(h, p["W_k"].T, p["b_k"])
    v = F.linear(h, p["W_v"].T, p["b_v"])
    batch, tokens, width = q.shape
    dh = width // heads
    if head_layout == "correct":  # head h owns features [h*dh, (h+1)*dh)
        split = lambda t: t.reshape(batch, tokens, heads, dh).permute(0, 2, 1, 3)  # noqa: E731
        merge = lambda o: o.permute(0, 2, 1, 3).reshape(batch, tokens, width)  # noqa: E731
    elif head_layout == "interleaved":  # head h owns features h, h+H, h+2H, ...
        split = lambda t: t.reshape(batch, tokens, dh, heads).permute(0, 3, 1, 2)  # noqa: E731
        merge = lambda o: o.permute(0, 2, 3, 1).reshape(batch, tokens, width)  # noqa: E731
    elif head_layout == "split_only":  # split like "unpermuted", merge correctly
        split = lambda t: t.reshape(batch, heads, tokens, dh)  # noqa: E731
        merge = lambda o: o.permute(0, 2, 1, 3).reshape(batch, tokens, width)  # noqa: E731
    else:  # "unpermuted": reshape without moving the token axis
        split = lambda t: t.reshape(batch, heads, tokens, dh)  # noqa: E731
        merge = lambda o: o.reshape(batch, tokens, width)  # noqa: E731
    qh, kh, vh = split(q), split(k), split(v)
    scale = math.sqrt(width if width_scale else dh)
    scores = qh @ kh.transpose(-1, -2) / scale
    if causal:
        idx = torch.arange(tokens)
        scores = scores.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
    out = merge(torch.softmax(scores, dim=-1) @ vh)
    return F.linear(out, p["W_o"].T, p["b_o"])


def _ffn(h, p, *, tanh_gelu=False):
    hidden = F.gelu(F.linear(h, p["W_1"].T, p["b_1"]), approximate="tanh" if tanh_gelu else "none")
    return F.linear(hidden, p["W_2"].T, p["b_2"])


def _block(x, p, heads, *, causal=True, residual_attn=True, residual_ffn=True, post=False,
           tanh_gelu=False, unbiased=False, eps_off=False, eps_outside=False,
           head_layout="correct", width_scale=False):  # fmt: skip
    """Reference block. Keyword flags switch on one known wrong variant each."""
    eps = 0.0 if eps_off else EPS

    def ln(u, g, b):
        return _ln(u, g, b, unbiased=unbiased, eps=eps, eps_outside=eps_outside)

    if post:  # original Vaswani placement: normalise the sum, not the sub-layer input
        a = _attn(x, p, heads, causal=causal, head_layout=head_layout, width_scale=width_scale)
        x1 = ln(x + a, p["ln1_gamma"], p["ln1_beta"])
        f = _ffn(x1, p, tanh_gelu=tanh_gelu)
        return ln(x1 + f, p["ln2_gamma"], p["ln2_beta"])
    h = ln(x, p["ln1_gamma"], p["ln1_beta"])
    a = _attn(h, p, heads, causal=causal, head_layout=head_layout, width_scale=width_scale)
    x_mid = x + a if residual_attn else a
    f = _ffn(ln(x_mid, p["ln2_gamma"], p["ln2_beta"]), p, tanh_gelu=tanh_gelu)
    return x_mid + f if residual_ffn else f


# (variant flags, first-line message) for each wrong implementation the tests can name.
MISTAKES = [
    (
        {"post": True},
        "post-norm placement: LayerNorm wraps the sum x + sublayer(x); "
        "normalise the sublayer input",
    ),
    (
        {"causal": False},
        "missing causal mask: later tokens leak into the attention of earlier positions",
    ),
    (
        {"residual_attn": False},
        "missing attention residual: the attention output must be added to x, not replace it",
    ),
    (
        {"residual_ffn": False},
        "missing feed-forward residual: add the FFN output to the stream, not in place of it",
    ),
    (
        {"head_layout": "unpermuted"},
        "heads read from the wrong axis: reshape to (B, T, H, d_head), then swap T and H",
    ),
    (
        {"head_layout": "split_only"},
        "heads read from the wrong axis: the split must be (B, T, H, d_head) then swap T and H",
    ),
    (
        {"head_layout": "interleaved"},
        "heads must own contiguous feature slices: split D into (H, d_head), not (d_head, H)",
    ),
    ({"tanh_gelu": True}, "GELU approximation used: the contract requires the exact erf form"),
    ({"unbiased": True}, "LayerNorm used unbiased variance: divide by D, not D - 1"),
    ({"eps_off": True}, "LayerNorm epsilon missing: add 1e-5 to the variance inside the sqrt"),
    (
        {"eps_outside": True},
        "LayerNorm epsilon placed outside the sqrt: add 1e-5 to the variance before the sqrt",
    ),
    (
        {"width_scale": True},
        "attention scores scaled by sqrt(D): scale by sqrt(d_head), the per-head width",
    ),
]


def _close(actual, expected):
    return actual.shape == expected.shape and torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)


def _run(x, p, heads, *, pass_through=False):
    """Call the learner on copies, so an in-place write cannot corrupt the oracle's inputs.

    Only the mutation test passes the original tensors, to observe any write into them.
    """
    if pass_through:
        call_x, call_p = x, p
    else:
        call_x, call_p = x.clone(), {key: value.clone() for key, value in p.items()}
    try:
        out = pre_norm_block(call_x, call_p, heads)
    except Exception as exc:  # any crash on valid input is a contract failure
        pytest.fail(
            f"valid input was rejected or crashed ({type(exc).__name__}): check the contract"
        )
    if not torch.is_tensor(out):
        pytest.fail("return a tensor, not a Python object")
    if tuple(out.shape) != tuple(x.shape):
        pytest.fail("output shape must equal the input shape (B, T, D)")
    if out.dtype != x.dtype:
        pytest.fail("output dtype must match x")
    if not torch.isfinite(out).all():
        pytest.fail("non-finite output: a query row has every score masked; keep the diagonal")
    return out


def _assert_matches(out, x, p, heads, expected=None):
    expected = _block(x, p, heads) if expected is None else expected
    if _close(out, expected):
        return
    for variant, message in MISTAKES:
        if _close(out, _block(x, p, heads, **variant)):
            pytest.fail(message)
    pytest.fail(
        "output differs from the pre-norm reference: check the order of LN, attention and residuals"
    )


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a value instead of raising ValueError")


@pytest.mark.parametrize(
    "batch, tokens, width, heads, ff",
    [(1, 1, 4, 1, 6), (2, 5, 8, 2, 12), (3, 4, 6, 3, 5), (2, 7, 12, 4, 20), (1, 3, 2, 2, 1)],
    ids=["single-token", "two-heads", "three-heads-odd-ff", "longer", "narrow"],
)
def test_matches_pre_norm_reference_on_seeded_batches(batch, tokens, width, heads, ff):
    gen = torch.Generator().manual_seed(batch * 100 + tokens)
    x = torch.randn(batch, tokens, width, generator=gen, dtype=torch.float64)
    p = _random_params(width, ff, gen)
    _assert_matches(_run(x, p, heads), x, p, heads)


def test_matches_torch_transformer_encoder_layer_with_same_weights():
    gen = torch.Generator().manual_seed(7)
    batch, tokens, width, heads, ff = 2, 4, 8, 2, 16
    x = torch.randn(batch, tokens, width, generator=gen, dtype=torch.float64)
    p = _random_params(width, ff, gen)
    layer = torch.nn.TransformerEncoderLayer(
        width, heads, ff, dropout=0.0, activation="gelu", layer_norm_eps=EPS,
        batch_first=True, norm_first=True,
    ).double()  # fmt: skip
    with torch.no_grad():
        attn = layer.self_attn
        attn.in_proj_weight.copy_(torch.cat([p["W_q"].T, p["W_k"].T, p["W_v"].T]))
        attn.in_proj_bias.copy_(torch.cat([p["b_q"], p["b_k"], p["b_v"]]))
        attn.out_proj.weight.copy_(p["W_o"].T)
        attn.out_proj.bias.copy_(p["b_o"])
        layer.linear1.weight.copy_(p["W_1"].T)
        layer.linear1.bias.copy_(p["b_1"])
        layer.linear2.weight.copy_(p["W_2"].T)
        layer.linear2.bias.copy_(p["b_2"])
        for norm, prefix in ((layer.norm1, "ln1"), (layer.norm2, "ln2")):
            norm.weight.copy_(p[f"{prefix}_gamma"])
            norm.bias.copy_(p[f"{prefix}_beta"])
        causal = torch.ones(tokens, tokens, dtype=torch.bool).triu(diagonal=1)
        expected = layer(x, src_mask=causal)
    _assert_matches(_run(x, p, heads), x, p, heads, expected=expected)


def test_future_tokens_do_not_change_earlier_outputs():
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(2, 5, 6, generator=gen, dtype=torch.float64)
    p = _random_params(6, 10, gen)
    before = _run(x, p, 2)
    changed = x.clone()
    changed[:, -1] += 5.0
    after = _run(changed, p, 2)
    if not torch.allclose(before[:, :-1], after[:, :-1], atol=ATOL, rtol=RTOL):
        pytest.fail(
            "future token changed earlier outputs: the causal mask is missing or misaligned"
        )


def test_zero_sublayer_weights_leave_the_residual_stream_unchanged():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 3, 4, generator=gen, dtype=torch.float64)
    p = _random_params(4, 6, gen)
    for key in ("W_o", "b_o", "W_2", "b_2"):
        p[key] = torch.zeros_like(p[key])
    out = _run(x, p, 2)
    if not torch.allclose(out, x, atol=ATOL, rtol=RTOL):
        pytest.fail(
            "with zero output projections the block must return x: check the residual additions"
        )


def test_batch_rows_are_processed_independently():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, 4, 6, generator=gen, dtype=torch.float64)
    p = _random_params(6, 8, gen)
    together = _run(x, p, 3)
    for row in range(x.shape[0]):
        alone = _run(x[row : row + 1], p, 3)
        if not torch.allclose(together[row : row + 1], alone, atol=ATOL, rtol=RTOL):
            pytest.fail("batch rows interact: a reshape or a mask mixes examples in the batch")


@pytest.mark.parametrize("scale", [1e4, 1e-4], ids=["large-magnitude", "small-magnitude"])
def test_extreme_input_scales_match_the_reference(scale):
    gen = torch.Generator().manual_seed(13)
    x = scale * torch.randn(2, 4, 6, generator=gen, dtype=torch.float64)
    p = _random_params(6, 9, gen)
    _assert_matches(_run(x, p, 2), x, p, 2)


def test_float32_inputs_return_float32_close_to_float64_reference():
    gen = torch.Generator().manual_seed(17)
    x64 = torch.randn(2, 4, 8, generator=gen, dtype=torch.float64)
    p64 = _random_params(8, 12, gen)
    x32 = x64.float()
    p32 = {k: v.float() for k, v in p64.items()}
    out = _run(x32, p32, 2)
    if out.dtype != torch.float32:
        pytest.fail("float32 input must return float32")
    # float32 has about 7 significant digits; 1e-4 absolute is about 1000 ulp at these magnitudes.
    if not torch.allclose(out.double(), _block(x64, p64, 2), atol=1e-4, rtol=1e-4):
        pytest.fail("float32 result drifts from the reference beyond single-precision round-off")


def test_inputs_and_parameters_are_not_mutated():
    gen = torch.Generator().manual_seed(19)
    x = torch.randn(2, 3, 4, generator=gen, dtype=torch.float64)
    p = _random_params(4, 5, gen)
    x_before, p_before = x.clone(), copy.deepcopy(p)
    _run(x, p, 2, pass_through=True)
    if not torch.equal(x, x_before):
        pytest.fail("the input x was modified: write into new tensors, not into x")
    for key in p:
        if not torch.equal(p[key], p_before[key]):
            pytest.fail("a parameter tensor was modified in place: never write into params")


def _params_with(override, *, d=4, ff=6):
    gen = torch.Generator().manual_seed(23)
    p = _random_params(d, ff, gen)
    p.update(override)
    return p


@pytest.mark.parametrize(
    "case",
    ["heads-do-not-divide", "zero-heads", "bool-heads", "missing-key", "extra-key",
     "wrong-projection-shape", "wrong-feed-forward-shape", "two-dimensional-x",
     "zero-tokens", "integer-x", "param-dtype-mismatch", "param-not-a-tensor"],
)  # fmt: skip
def test_invalid_inputs_raise_value_error(case):
    x = torch.ones(2, 3, 4, dtype=torch.float64)
    p = _params_with({})
    heads = 2
    if case == "heads-do-not-divide":
        heads = 3
    elif case == "zero-heads":
        heads = 0
    elif case == "bool-heads":
        heads = True
    elif case == "missing-key":
        del p["W_o"]
    elif case == "extra-key":
        p["dropout"] = torch.zeros(1, dtype=torch.float64)
    elif case == "wrong-projection-shape":
        p["W_q"] = torch.zeros(4, 3, dtype=torch.float64)
    elif case == "wrong-feed-forward-shape":
        p["W_2"] = torch.zeros(5, 4, dtype=torch.float64)
    elif case == "two-dimensional-x":
        x = torch.ones(3, 4, dtype=torch.float64)
    elif case == "zero-tokens":
        x = torch.ones(2, 0, 4, dtype=torch.float64)
    elif case == "integer-x":
        x = torch.ones(2, 3, 4, dtype=torch.int64)
    elif case == "param-dtype-mismatch":
        p["b_1"] = p["b_1"].float()
    elif case == "param-not-a-tensor":
        p["b_2"] = p["b_2"].tolist()
    _expect_value_error(lambda: pre_norm_block(x, p, heads), f"invalid input ({case}) was accepted")


def test_non_tensor_x_is_rejected_as_value_error():
    x = [[[1.0, 2.0, 3.0, 4.0], [0.5, 0.25, 0.1, 0.2], [1.0, 1.0, 1.0, 1.0]]] * 2
    _expect_value_error(
        lambda: pre_norm_block(x, _params_with({}), 2),
        "x that is not a tensor must be rejected with ValueError",
    )


def test_integer_x_with_integer_params_is_rejected_as_value_error():
    x = torch.ones(2, 3, 4, dtype=torch.int64)
    p = {key: torch.ones(value.shape, dtype=torch.int64) for key, value in _params_with({}).items()}
    _expect_value_error(
        lambda: pre_norm_block(x, p, 2),
        "an integer x must be rejected with ValueError even when the parameters match its dtype",
    )


def test_one_dimensional_feed_forward_weight_is_rejected_as_value_error():
    x = torch.ones(2, 3, 4, dtype=torch.float64)
    p = _params_with({"W_1": torch.zeros(4, dtype=torch.float64)})
    _expect_value_error(
        lambda: pre_norm_block(x, p, 2),
        "W_1 must be a two-dimensional (D, F) matrix; a one-dimensional W_1 must raise ValueError",
    )
