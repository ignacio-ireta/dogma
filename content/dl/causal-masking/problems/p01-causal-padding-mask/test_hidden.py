# Hidden tests: exhaustive small-shape oracle (tril with offset), independent padding AND, decode
# alignment, torch SDPA agreement on square masks, fully masked rows, dtype, sizes and mutation.
import functools

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - oracle only
from submission import causal_padding_mask


def checked(call):
    try:
        out = call()
    except Exception:
        pytest.fail(
            "valid input raised an exception: honor the documented interface", pytrace=False
        )
    if not isinstance(out, torch.Tensor):
        pytest.fail("return a boolean tensor", pytrace=False)
    return out


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "invalid input must raise ValueError, not another exception type", pytrace=False
        )
    pytest.fail("invalid input must raise ValueError but returned a mask", pytrace=False)


def oracle_causal(L, S):
    # Bottom-right alignment expressed as a lower triangle shifted by the offset S - L.
    return torch.ones(L, S, dtype=torch.bool).tril(diagonal=S - L)


def test_causal_part_matches_tril_oracle_for_every_small_shape():
    for L in range(0, 7):
        for S in range(0, 7):
            got = checked(functools.partial(causal_padding_mask, L, S))
            if tuple(got.shape) != (L, S):
                pytest.fail("causal mask must have shape (query_len, key_len)", pytrace=False)
            if not torch.equal(got, oracle_causal(L, S)):
                pytest.fail("causal mask disagrees with the bottom-right rule", pytrace=False)


@pytest.mark.parametrize("seed", range(12), ids=list("abcdefghijkl"))
def test_padding_is_anded_along_keys_with_causal_rule(seed):
    g = torch.Generator().manual_seed(seed)
    L, S, N = (int(x) for x in torch.randint(1, 7, (3,), generator=g))
    key_valid = torch.rand(N, S, generator=g) > 0.35
    got = checked(functools.partial(causal_padding_mask, L, S, key_valid))
    expected = oracle_causal(L, S).unsqueeze(0) & key_valid.unsqueeze(1)
    if tuple(got.shape) != (N, L, S):
        pytest.fail("padded mask must have shape (N, query_len, key_len)", pytrace=False)
    if not torch.equal(got, expected):
        pytest.fail(
            "padding must remove whole key columns for each batch row (AND with causal)",
            pytrace=False,
        )


def test_single_query_decode_step_sees_every_valid_key():
    key_valid = torch.tensor([[True, True, False, True, True]])
    got = checked(functools.partial(causal_padding_mask, 1, 5, key_valid))
    if not torch.equal(got, key_valid.unsqueeze(1)):
        pytest.fail("a single new query must see every valid cached key", pytrace=False)


def test_cached_queries_are_not_top_left_aligned():
    # torch's is_causal is top-left for non-square masks; the cache case needs bottom-right.
    got = checked(functools.partial(causal_padding_mask, 2, 5))
    top_left = torch.ones(2, 5, dtype=torch.bool).tril(diagonal=0)
    if torch.equal(got, top_left):
        pytest.fail(
            "used top-left alignment: with a KV cache the queries are the last keys", pytrace=False
        )
    if not torch.equal(got, oracle_causal(2, 5)):
        pytest.fail("causal rows for cached queries are misaligned", pytrace=False)


def test_square_mask_reproduces_torch_is_causal_attention():
    g = torch.Generator().manual_seed(7)
    q = torch.randn(4, 3, generator=g, dtype=torch.float64)
    k = torch.randn(4, 3, generator=g, dtype=torch.float64)
    v = torch.randn(4, 2, generator=g, dtype=torch.float64)
    mask = checked(functools.partial(causal_padding_mask, 4, 4))
    via_mask = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    via_flag = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if not torch.allclose(via_mask, via_flag, atol=1e-9, rtol=1e-9):
        pytest.fail("square causal mask does not reproduce is_causal attention", pytrace=False)


def test_fully_masked_rows_are_all_false_and_never_raise():
    got = checked(functools.partial(causal_padding_mask, 4, 2))
    expected_rows = torch.tensor([False, False, True, True])
    if not torch.equal(got.any(dim=-1), expected_rows):
        pytest.fail(
            "rows with no allowed key must be all False, and no other row may change", pytrace=False
        )
    key_valid = torch.tensor([[False, False, False], [True, True, False]])
    padded = checked(functools.partial(causal_padding_mask, 2, 3, key_valid))
    if padded[0].any():
        pytest.fail(
            "an all-padding sequence must give an all-False mask, not True entries", pytrace=False
        )


def test_output_is_boolean_with_contract_shapes():
    unpadded = checked(functools.partial(causal_padding_mask, 2, 3))
    if unpadded.dtype != torch.bool:
        pytest.fail("mask must be boolean, not an integer or float additive mask", pytrace=False)
    empty_batch = checked(
        functools.partial(causal_padding_mask, 2, 3, torch.zeros(0, 3, dtype=torch.bool))
    )
    if tuple(empty_batch.shape) != (0, 2, 3):
        pytest.fail("zero-size batch must return shape (0, query_len, key_len)", pytrace=False)


def test_zero_sized_queries_or_keys_give_empty_masks():
    if tuple(checked(functools.partial(causal_padding_mask, 0, 3)).shape) != (0, 3):
        pytest.fail("zero queries must give an empty (0, key_len) mask", pytrace=False)
    if tuple(checked(functools.partial(causal_padding_mask, 2, 0)).shape) != (2, 0):
        pytest.fail("zero keys must give an empty (query_len, 0) mask", pytrace=False)


def test_invalid_sizes_rejected():
    rejects(lambda: causal_padding_mask(-1, 3))
    rejects(lambda: causal_padding_mask(2, -3))
    rejects(lambda: causal_padding_mask(2.0, 3))  # float size
    rejects(lambda: causal_padding_mask(True, 3))  # bool is not a size


def test_key_valid_must_be_boolean_and_two_dimensional():
    rejects(lambda: causal_padding_mask(2, 3, torch.ones(1, 3, dtype=torch.int64)))
    rejects(lambda: causal_padding_mask(2, 3, torch.ones(1, 4, dtype=torch.bool)))
    rejects(lambda: causal_padding_mask(2, 3, torch.ones(3, dtype=torch.bool)))


def test_key_valid_is_not_mutated_and_output_does_not_alias_it():
    key_valid = torch.tensor([[True, False, False], [True, True, False]])
    before = key_valid.clone()
    out = checked(functools.partial(causal_padding_mask, 2, 3, key_valid))
    out[...] = False
    if not torch.equal(key_valid, before):
        pytest.fail("key_valid must not be mutated; return a new tensor", pytrace=False)


def test_large_sizes_count_allowed_pairs_exactly():
    L, S = 64, 200
    got = checked(functools.partial(causal_padding_mask, L, S))
    # Query i allows S - L + i + 1 keys when that count is positive.
    expected_count = sum(max(0, S - L + i + 1) for i in range(L))
    if int(got.sum()) != expected_count:
        pytest.fail("allowed-pair count is wrong for a large decode-sized mask", pytrace=False)


def test_key_valid_with_fewer_than_key_len_columns_is_rejected():
    # Contract: key_valid must have exactly S columns. Two columns for S = 3 is invalid.
    rejects(lambda: causal_padding_mask(2, 3, torch.ones(1, 2, dtype=torch.bool)))
    rejects(lambda: causal_padding_mask(1, 3, torch.ones(2, 2, dtype=torch.bool)))
