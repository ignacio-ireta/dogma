# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import pytest
import torch
from submission import cached_attention_step as _learner


def _call(*args, **kwargs):
    try:
        return _learner(*args, **kwargs)
    except ValueError:
        pytest.fail("valid arguments were rejected with ValueError: check the argument rules")
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _expect_value_error(*args, **kwargs):
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

    return mat(), mat(), mat(), mat()


def _full_causal(x, W_q, W_k, W_v, W_o, heads):  # noqa: PLR0917 (oracle takes the weight set)
    """Oracle: torch.nn.MultiheadAttention with the same weights, no bias, over the whole prefix.

    Its attn_mask is True where a query may NOT attend, so the upper triangle is blocked.
    """
    d = x.shape[-1]
    mha = torch.nn.MultiheadAttention(d, heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([W_q, W_k, W_v], dim=0))
        mha.out_proj.weight.copy_(W_o)
    mha.eval()
    L = x.shape[1]
    blocked = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    out, _ = mha(x, x, x, attn_mask=blocked, need_weights=False)
    return out


def _run_chunks(x, weights, heads, chunk_sizes):
    """Feed x through the learner in the given chunk sizes; return the concatenated outputs."""
    W_q, W_k, W_v, W_o = weights
    cache = None
    outs = []
    start = 0
    for size in chunk_sizes:
        chunk = x[:, start : start + size]
        out, cache = _call(chunk, cache, W_q, W_k, W_v, W_o, heads)
        outs.append(out)
        start += size
    return torch.cat(outs, dim=1), cache


def test_decode_steps_match_full_recomputation_single_head():
    gen = torch.Generator().manual_seed(31)
    d, heads, steps = 6, 1, 6
    x = torch.randn(2, steps, d, generator=gen, dtype=torch.float64)
    weights = _weights(gen, d)
    got, _ = _run_chunks(x, weights, heads, [1] * steps)
    want = _full_causal(x, *weights, heads)
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "single-head decode differs from full causal recomputation: "
            "check the cache append and the offset"
        ),
    )


@pytest.mark.parametrize("heads", [2, 4], ids=["two_heads", "four_heads"])
def test_decode_steps_match_full_recomputation_multi_head(heads):
    gen = torch.Generator().manual_seed(40 + heads)
    d, steps = 8, 5
    x = torch.randn(3, steps, d, generator=gen, dtype=torch.float64)
    weights = _weights(gen, d)
    got, _ = _run_chunks(x, weights, heads, [1] * steps)
    want = _full_causal(x, *weights, heads)
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "multi-head decode differs from full recomputation: check the "
            "head split, merge and 1/sqrt(d_k)"
        ),
    )


def test_prefill_chunk_then_single_steps_match_full_recomputation():
    gen = torch.Generator().manual_seed(52)
    d, heads = 8, 2
    x = torch.randn(2, 6, d, generator=gen, dtype=torch.float64)
    weights = _weights(gen, d)
    got, _ = _run_chunks(x, weights, heads, [3, 1, 1, 1])
    want = _full_causal(x, *weights, heads)
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "prefill chunk then decode differs from full recomputation: "
            "check the mask for a multi-token chunk"
        ),
    )


def test_chunk_of_several_tokens_after_cache_uses_the_bottom_right_offset():
    gen = torch.Generator().manual_seed(61)
    d, heads = 6, 2
    x = torch.randn(1, 7, d, generator=gen, dtype=torch.float64)
    weights = _weights(gen, d)
    got, _ = _run_chunks(x, weights, heads, [2, 3, 2])
    want = _full_causal(x, *weights, heads)
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "a multi-token chunk after a cache ignored the offset: query "
            "i must see keys up to S + i"
        ),
    )


def test_one_chunk_equals_token_by_token_decoding():
    gen = torch.Generator().manual_seed(70)
    d, heads = 8, 4
    x = torch.randn(2, 5, d, generator=gen, dtype=torch.float64)
    weights = _weights(gen, d)
    all_at_once, _ = _run_chunks(x, weights, heads, [5])
    one_by_one, _ = _run_chunks(x, weights, heads, [1, 1, 1, 1, 1])
    torch.testing.assert_close(
        all_at_once,
        one_by_one,
        atol=1e-9,
        rtol=1e-9,
        msg="chunked and token-by-token outputs disagree: the mask for chunks is wrong",
    )


def test_cache_grows_by_exactly_the_new_tokens_and_keeps_earlier_positions():
    gen = torch.Generator().manual_seed(80)
    d, heads = 6, 2
    d_k = d // heads
    x = torch.randn(2, 4, d, generator=gen, dtype=torch.float64)
    W_q, W_k, W_v, W_o = _weights(gen, d)
    _, cache = _call(x[:, :3], None, W_q, W_k, W_v, W_o, heads)
    K_old, V_old = cache[0].clone(), cache[1].clone()
    assert cache[0].shape == (2, heads, 3, d_k), (
        "after a 3-token prefill the cache must hold 3 positions"
    )
    _, cache2 = _call(x[:, 3:], cache, W_q, W_k, W_v, W_o, heads)
    assert cache2[0].shape == (2, heads, 4, d_k), (
        "the cache must grow by exactly the new token count"
    )
    assert torch.equal(cache2[0][:, :, :3], K_old), (
        "earlier cached keys were replaced instead of kept"
    )
    assert torch.equal(cache2[1][:, :, :3], V_old), (
        "earlier cached values were replaced instead of kept"
    )


def test_input_cache_is_not_mutated():
    gen = torch.Generator().manual_seed(81)
    d, heads = 6, 3
    x = torch.randn(2, 3, d, generator=gen, dtype=torch.float64)
    W_q, W_k, W_v, W_o = _weights(gen, d)
    _, cache = _call(x[:, :2], None, W_q, W_k, W_v, W_o, heads)
    before = (cache[0].clone(), cache[1].clone())
    _call(x[:, 2:], cache, W_q, W_k, W_v, W_o, heads)
    assert torch.equal(cache[0], before[0]) and torch.equal(cache[1], before[1]), (
        "the input cache was modified: build the extended cache in new tensors"
    )


def test_input_and_weights_are_not_mutated():
    gen = torch.Generator().manual_seed(82)
    d, heads = 4, 2
    x = torch.randn(1, 2, d, generator=gen, dtype=torch.float64)
    weights = _weights(gen, d)
    x_before = x.clone()
    w_before = [w.clone() for w in weights]
    _call(x, None, *weights, heads)
    assert torch.equal(x, x_before), "x_new was modified in place"
    assert all(torch.equal(w, b) for w, b in zip(weights, w_before, strict=True)), (
        "a weight was modified in place"
    )


def test_empty_zero_length_cache_matches_none():
    gen = torch.Generator().manual_seed(83)
    d, heads = 6, 2
    x = torch.randn(2, 3, d, generator=gen, dtype=torch.float64)
    W = _weights(gen, d)
    empty = (torch.zeros(2, heads, 0, d // heads, dtype=torch.float64),) * 2
    a, _ = _call(x, None, *W, heads)
    b, _ = _call(x, empty, *W, heads)
    torch.testing.assert_close(
        b, a, atol=1e-12, rtol=0, msg="an empty cache must behave like no cache"
    )


def test_batch_rows_are_independent():
    gen = torch.Generator().manual_seed(84)
    d, heads = 6, 2
    x = torch.randn(3, 4, d, generator=gen, dtype=torch.float64)
    W = _weights(gen, d)
    together, _ = _run_chunks(x, W, heads, [2, 1, 1])
    alone = torch.cat([_run_chunks(x[b : b + 1], W, heads, [2, 1, 1])[0] for b in range(3)], dim=0)
    torch.testing.assert_close(
        together, alone, atol=1e-9, rtol=1e-9, msg="batch rows influence each other"
    )


def test_large_and_small_magnitudes_match_full_recomputation():
    gen = torch.Generator().manual_seed(85)
    d, heads = 6, 2
    for scale in (1e-6, 1e3):
        x = torch.randn(1, 4, d, generator=gen, dtype=torch.float64) * scale
        W = _weights(gen, d, scale=0.5)
        got, _ = _run_chunks(x, W, heads, [1, 1, 1, 1])
        want = _full_causal(x, *W, heads)
        torch.testing.assert_close(
            got,
            want,
            atol=1e-9 * max(1.0, scale),
            rtol=1e-9,
            msg="magnitude-dependent error: check the scale and whether the softmax is stable",
        )


def test_output_has_input_shape_and_float64_dtype():
    gen = torch.Generator().manual_seed(86)
    d, heads = 8, 4
    x = torch.randn(2, 3, d, generator=gen, dtype=torch.float64)
    out, cache = _call(x, None, *_weights(gen, d), heads)
    assert out.shape == (2, 3, d), "output must be (B, T, d)"
    assert out.dtype == torch.float64 and cache[0].dtype == torch.float64, "dtype must be float64"


def test_zero_new_tokens_raises_value_error():
    gen = torch.Generator().manual_seed(87)
    _expect_value_error(torch.zeros(1, 0, 4, dtype=torch.float64), None, *_weights(gen, 4), 2)


def test_num_heads_that_does_not_divide_d_raises_value_error():
    gen = torch.Generator().manual_seed(88)
    _expect_value_error(torch.zeros(1, 2, 6, dtype=torch.float64), None, *_weights(gen, 6), 4)


def test_bool_num_heads_raises_value_error():
    gen = torch.Generator().manual_seed(89)
    _expect_value_error(torch.zeros(1, 2, 4, dtype=torch.float64), None, *_weights(gen, 4), True)


def test_float32_input_raises_value_error():
    gen = torch.Generator().manual_seed(90)
    _expect_value_error(torch.zeros(1, 2, 4, dtype=torch.float32), None, *_weights(gen, 4), 2)


def test_wrong_weight_shape_raises_value_error():
    gen = torch.Generator().manual_seed(91)
    W_q, W_k, W_v, W_o = _weights(gen, 4)
    _expect_value_error(
        torch.zeros(1, 2, 4, dtype=torch.float64), None, W_q[:, :3], W_k, W_v, W_o, 2
    )


def test_cache_with_wrong_head_layout_raises_value_error():
    gen = torch.Generator().manual_seed(92)
    d, heads = 6, 2
    bad_cache = (torch.zeros(1, 2, 3, d, dtype=torch.float64),) * 2  # d_k should be d // heads
    _expect_value_error(
        torch.zeros(1, 1, d, dtype=torch.float64), bad_cache, *_weights(gen, d), heads
    )


def test_cache_batch_mismatch_raises_value_error():
    gen = torch.Generator().manual_seed(93)
    d, heads = 4, 2
    bad_cache = (torch.zeros(3, heads, 1, d // heads, dtype=torch.float64),) * 2
    _expect_value_error(
        torch.zeros(1, 1, d, dtype=torch.float64), bad_cache, *_weights(gen, d), heads
    )


def test_cache_of_wrong_type_raises_value_error():
    gen = torch.Generator().manual_seed(94)
    d = 4
    _expect_value_error(
        torch.zeros(1, 1, d, dtype=torch.float64), torch.zeros(1), *_weights(gen, d), 2
    )


def test_zero_batch_raises_value_error():
    gen = torch.Generator().manual_seed(95)
    _expect_value_error(torch.zeros(0, 1, 4, dtype=torch.float64), None, *_weights(gen, 4), 2)


def test_cache_that_is_not_a_pair_raises_value_error():
    gen = torch.Generator().manual_seed(96)
    _expect_value_error(torch.zeros(1, 1, 4, dtype=torch.float64), 3, *_weights(gen, 4), 2)


def test_cache_keys_and_values_of_different_lengths_raise_value_error():
    gen = torch.Generator().manual_seed(97)
    K = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    V = torch.zeros(1, 2, 1, 2, dtype=torch.float64)
    _expect_value_error(torch.zeros(1, 1, 4, dtype=torch.float64), (K, V), *_weights(gen, 4), 2)


def test_cache_in_another_dtype_raises_value_error():
    gen = torch.Generator().manual_seed(98)
    K = torch.zeros(1, 2, 1, 2, dtype=torch.float32)
    V = torch.zeros(1, 2, 1, 2, dtype=torch.float32)
    _expect_value_error(torch.zeros(1, 1, 4, dtype=torch.float64), (K, V), *_weights(gen, 4), 2)


def test_cache_that_is_not_four_dimensional_raises_value_error():
    gen = torch.Generator().manual_seed(99)
    K = torch.zeros(1, 2, 2, dtype=torch.float64)
    V = torch.zeros(1, 2, 2, dtype=torch.float64)
    _expect_value_error(torch.zeros(1, 1, 4, dtype=torch.float64), (K, V), *_weights(gen, 4), 2)
