# Hidden tests for grouped_query_attention. The oracle is torch.nn.functional's
# scaled-dot-product attention, fed with an explicit key/value index per query head that the
# contract states. The one-token and mask cases are hand-checked. Failure messages name the
# kind of mistake and never show hidden inputs, shapes or expected values.
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import grouped_query_attention
from torch.nn.functional import scaled_dot_product_attention


def _raises(expected, call, mistake):
    try:
        call()
    except expected:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


CONFIGS = {
    "multi-query": (4, 1, 2),
    "grouped-two-of-four": (4, 2, 3),
    "grouped-two-of-six": (6, 2, 2),
    "grouped-three-of-six": (6, 3, 2),
    "full-multi-head": (4, 4, 2),
}


def _random_case(  # noqa: PLR0917
    seed: int, B: int, L: int, n_heads: int, n_kv_heads: int, d_head: int
):
    g = torch.Generator().manual_seed(seed)
    d = n_heads * d_head
    kv = n_kv_heads * d_head
    X = torch.randn(B, L, d, generator=g, dtype=torch.float64)
    W_q = torch.randn(d, d, generator=g, dtype=torch.float64)
    W_k = torch.randn(kv, d, generator=g, dtype=torch.float64)
    W_v = torch.randn(kv, d, generator=g, dtype=torch.float64)
    W_o = torch.randn(d, d, generator=g, dtype=torch.float64)
    return X, W_q, W_k, W_v, W_o


def _oracle(X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads, mask=None):  # noqa: PLR0917
    """Reference: query head h uses key/value head h // group_size, written out as an index list."""
    B, L, d = X.shape
    d_head = d // n_heads
    group_size = n_heads // n_kv_heads
    kv_index = torch.tensor([h // group_size for h in range(n_heads)])
    Q = (X @ W_q.T).reshape(B, L, n_heads, d_head).transpose(1, 2)
    K = (X @ W_k.T).reshape(B, L, n_kv_heads, d_head).transpose(1, 2).index_select(1, kv_index)
    V = (X @ W_v.T).reshape(B, L, n_kv_heads, d_head).transpose(1, 2).index_select(1, kv_index)
    context = scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    return context.transpose(1, 2).reshape(B, L, d) @ W_o.T


def _assert_matches(got: torch.Tensor, want: torch.Tensor, kind: str) -> None:
    # Shape and dtype guards come first, so a transposed or downcast result gets a named message.
    assert tuple(got.shape) == tuple(want.shape), f"{kind}: output shape is not (B, L, d_model)"
    assert got.dtype == torch.float64, f"{kind}: output dtype must be float64"
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-9), (
        f"{kind}: values differ from the reference (check the head mapping, the 1/sqrt(d_head) "
        "scale, the mask, the nn.Linear weight layout and the output projection W_o)"
    )


def _call_or_report(fn, *args, **kwargs):
    """Return (value, None), or (None, description). Errors are reported after the try block."""
    try:
        return fn(*args, **kwargs), None
    except ValueError as exc:
        return None, f"raised ValueError on a valid input ({exc})"
    except Exception as exc:
        return None, f"raised {type(exc).__name__} on a valid input"


@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
def test_random_inputs_match_reference(config: str):
    n_heads, n_kv_heads, d_head = CONFIGS[config]
    for seed in range(3):
        X, W_q, W_k, W_v, W_o = _random_case(seed, 2, 5, n_heads, n_kv_heads, d_head)
        got, err = _call_or_report(
            grouped_query_attention, X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads
        )
        assert err is None, f"{config}: {err}"
        _assert_matches(
            got, _oracle(X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads), f"{config} grouping"
        )


def test_six_heads_two_groups_one_token_checks_group_order():
    # 6 query heads, 2 key/value heads, group_size 3: heads 0-2 read 3, heads 3-5 read 4.
    # The modulo mapping h % 2 would give [3, 4, 3, 4, 3, 4] instead.
    X = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype=torch.float64)
    eye = torch.eye(6, dtype=torch.float64)
    W_k = torch.zeros(2, 6, dtype=torch.float64)
    W_k[0, 0] = W_k[1, 1] = 1.0
    W_v = torch.zeros(2, 6, dtype=torch.float64)
    W_v[0, 2] = W_v[1, 3] = 1.0
    got, err = _call_or_report(grouped_query_attention, X, eye, W_k, W_v, eye, 6, 2)
    assert err is None, f"six-head case: {err}"
    want = torch.tensor([3.0, 3.0, 3.0, 4.0, 4.0, 4.0], dtype=torch.float64)
    assert torch.allclose(got[0, 0], want), (
        "head mapping wrong: query head h must read key/value head h // group_size"
    )


def test_one_token_with_three_key_value_heads_maps_contiguous_pairs():
    # 6 query heads, 3 key/value heads, group_size 2: key/value head g serves heads 2g, 2g+1.
    X = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype=torch.float64)
    eye = torch.eye(6, dtype=torch.float64)
    W_k = torch.zeros(3, 6, dtype=torch.float64)
    W_k[0, 0] = W_k[1, 1] = W_k[2, 2] = 1.0
    W_v = torch.zeros(3, 6, dtype=torch.float64)
    W_v[0, 3] = W_v[1, 4] = W_v[2, 5] = 1.0
    got, err = _call_or_report(grouped_query_attention, X, eye, W_k, W_v, eye, 6, 3)
    assert err is None, f"three-kv-head case: {err}"
    want = torch.tensor([4.0, 4.0, 5.0, 5.0, 6.0, 6.0], dtype=torch.float64)
    assert torch.allclose(got[0, 0], want), (
        "head mapping wrong: query heads 2g and 2g+1 must read key/value head g"
    )


def test_multi_query_equals_tiled_full_attention():
    # One shared key/value head is the same as n_heads full heads whose key and value projections
    # are copies of that one head.
    n_heads, d_head = 4, 2
    X, W_q, W_k, W_v, W_o = _random_case(5, 2, 4, n_heads, 1, d_head)
    got, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, n_heads, 1)
    assert err is None, f"multi-query case: {err}"
    want = _oracle(X, W_q, W_k.repeat(n_heads, 1), W_v.repeat(n_heads, 1), W_o, n_heads, n_heads)
    _assert_matches(got, want, "multi-query sharing")


def test_mask_true_means_allowed_and_all_true_mask_equals_no_mask():
    X, W_q, W_k, W_v, W_o = _random_case(9, 2, 4, 4, 2, 2)
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [True, False, True, True],
        ]
    )
    got, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2, mask)
    assert err is None, f"masked case: {err}"
    _assert_matches(got, _oracle(X, W_q, W_k, W_v, W_o, 4, 2, mask), "mask meaning")
    all_true = torch.ones(4, 4, dtype=torch.bool)
    no_mask, _ = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2)
    with_true, err2 = _call_or_report(
        grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2, all_true
    )
    assert err2 is None, f"all-true mask: {err2}"
    _assert_matches(with_true, no_mask, "all-True mask must match no mask")


def test_scale_uses_head_width_not_model_width():
    # d_model = 8 with 4 heads gives d_head = 2. The scale must be 1/sqrt(2), not 1/sqrt(8).
    X, W_q, W_k, W_v, W_o = _random_case(21, 1, 5, 4, 2, 2)
    X = X * 2.0
    got, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2)
    assert err is None, f"scale case: {err}"
    _assert_matches(got, _oracle(X, W_q, W_k, W_v, W_o, 4, 2), "attention scale 1/sqrt(d_head)")


def test_nonsymmetric_weights_use_x_times_transpose():
    X = torch.tensor([[[1.0, 2.0], [0.5, -1.0]]], dtype=torch.float64)
    W_q = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
    W_k = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    W_v = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    W_o = torch.tensor([[1.0, 4.0], [0.0, 2.0]], dtype=torch.float64)
    got, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 2, 1)
    assert err is None, f"layout case: {err}"
    _assert_matches(got, _oracle(X, W_q, W_k, W_v, W_o, 2, 1), "nn.Linear layout")


def test_fully_masked_query_row_raises_value_error():
    X, W_q, W_k, W_v, W_o = _random_case(3, 1, 3, 2, 1, 2)
    mask = torch.tensor([[True, False, False], [False, False, False], [True, True, True]])
    _raises(
        ValueError,
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, 2, 1, mask),
        "a fully masked query row must raise ValueError",
    )


def _problem_or_none(call) -> str | None:
    try:
        call()
    except ValueError:
        return None
    except Exception as exc:  # reported after the try block
        return f"raised {type(exc).__name__} instead of ValueError"
    return "returned a value instead of raising ValueError"


def test_full_width_key_projection_raises_value_error():
    # The most common mistake: K and V projected to d_model when n_kv_heads < n_heads.
    X, W_q, _, W_v, W_o = _random_case(4, 1, 3, 4, 2, 2)
    full_k = torch.randn(8, 8, dtype=torch.float64)
    problem = _problem_or_none(lambda: grouped_query_attention(X, W_q, full_k, W_v, W_o, 4, 2))
    assert problem is None, f"key projection width not checked: {problem}"


def test_invalid_head_counts_raise_value_error():
    X, W_q, W_k, W_v, W_o = _random_case(6, 1, 2, 4, 2, 2)
    checks = [
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, 4, 3),  # does not divide
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, 2, 4),  # more kv heads than heads
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, 0, 1),  # zero heads
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, 4, 0),  # zero kv heads
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, True, 1),  # bool is not a count
        lambda: grouped_query_attention(X, W_q, W_k, W_v, W_o, 3, 1),  # does not divide d_model
    ]
    for call in checks:
        problem = _problem_or_none(call)
        assert problem is None, f"head count rule not enforced: {problem}"


def test_bad_mask_dtype_or_shape_raises_value_error():
    X, W_q, W_k, W_v, W_o = _random_case(8, 1, 3, 2, 1, 2)
    float_mask = torch.ones(3, 3, dtype=torch.float64)
    wide_mask = torch.ones(3, 4, dtype=torch.bool)
    for mask in (float_mask, wide_mask):
        problem = _problem_or_none(
            lambda m=mask: grouped_query_attention(X, W_q, W_k, W_v, W_o, 2, 1, m)
        )
        assert problem is None, f"mask must be a bool (L, L) tensor: {problem}"


def test_batch_items_are_independent_of_each_other():
    X, W_q, W_k, W_v, W_o = _random_case(12, 3, 4, 4, 2, 2)
    batch, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2)
    assert err is None, f"batch case: {err}"
    for b in range(3):
        single, err_b = _call_or_report(
            grouped_query_attention, X[b : b + 1], W_q, W_k, W_v, W_o, 4, 2
        )
        assert err_b is None, f"single item: {err_b}"
        _assert_matches(
            batch[b : b + 1], single, "batch independence (items mixed across the batch)"
        )


def test_large_and_small_input_magnitudes_match_reference():
    for scale in (1e2, 1e-2):
        X, W_q, W_k, W_v, W_o = _random_case(15, 1, 4, 4, 2, 2)
        X = X * scale
        got, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2)
        assert err is None, f"magnitude case: {err}"
        _assert_matches(got, _oracle(X, W_q, W_k, W_v, W_o, 4, 2), "large or small input magnitude")


def test_inputs_are_not_mutated():
    X, W_q, W_k, W_v, W_o = _random_case(16, 1, 3, 4, 2, 2)
    copies = [t.clone() for t in (X, W_q, W_k, W_v, W_o)]
    _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, 4, 2)
    for before, after in zip(copies, (X, W_q, W_k, W_v, W_o), strict=True):
        assert torch.equal(before, after), "an input tensor was modified in place"


@settings(max_examples=40, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    B=st.integers(min_value=1, max_value=3),
    L=st.integers(min_value=1, max_value=4),
    heads=st.sampled_from([(1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (6, 2), (6, 3)]),
)
def test_property_random_shapes_match_reference(seed: int, B: int, L: int, heads: tuple[int, int]):
    n_heads, n_kv_heads = heads
    X, W_q, W_k, W_v, W_o = _random_case(seed, B, L, n_heads, n_kv_heads, 2)
    got, err = _call_or_report(grouped_query_attention, X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads)
    assert err is None, f"random shape case: {err}"
    _assert_matches(
        got, _oracle(X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads), "random shape grouping"
    )


def test_n_heads_not_dividing_d_model_raises_value_error():
    # d_model = 4 with 3 heads: the weights below have the shapes a learner would pass for
    # d_head = 4 // 3, so only the divisibility rule can reject this call.
    X = torch.arange(1, 9, dtype=torch.float64).reshape(1, 2, 4)
    eye = torch.eye(4, dtype=torch.float64)
    W_k = torch.zeros(1, 4, dtype=torch.float64)
    W_v = torch.zeros(1, 4, dtype=torch.float64)
    problem = _problem_or_none(lambda: grouped_query_attention(X, eye, W_k, W_v, eye, 3, 1))
    assert problem is None, (
        f"n_heads must divide d_model, but this call was not rejected with ValueError: {problem}"
    )


def test_n_kv_heads_not_dividing_n_heads_raises_value_error():
    # 4 query heads and 3 key/value heads with d_model = 8: the weights have the shapes a learner
    # would pass for 3 key/value heads, so only the divisibility rule can reject this call.
    X = torch.arange(1, 17, dtype=torch.float64).reshape(1, 2, 8)
    eye = torch.eye(8, dtype=torch.float64)
    W_k = torch.zeros(6, 8, dtype=torch.float64)
    W_v = torch.zeros(6, 8, dtype=torch.float64)
    problem = _problem_or_none(lambda: grouped_query_attention(X, eye, W_k, W_v, eye, 4, 3))
    assert problem is None, (
        f"n_kv_heads must divide n_heads, but this call was not rejected with ValueError: {problem}"
    )


def test_bool_head_count_raises_even_when_shapes_fit():
    # d_model = 2. With n_heads = True (1) and n_kv_heads = 1, every weight has shape (2, 2),
    # so the weight-shape rule cannot be what rejects these calls: only the bool rule can.
    X = torch.ones(1, 2, 2, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    for call in (
        lambda: grouped_query_attention(X, eye, eye, eye, eye, True, 1),
        lambda: grouped_query_attention(X, eye, eye, eye, eye, 1, True),
    ):
        problem = _problem_or_none(call)
        assert problem is None, (
            "a bool head count must be rejected with ValueError, as the contract says"
        )
