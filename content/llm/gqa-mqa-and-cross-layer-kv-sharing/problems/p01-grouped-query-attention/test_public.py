# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import grouped_query_attention


def _one_token_case():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.float64)
    w_k = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=torch.float64)
    w_v = torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]], dtype=torch.float64)
    return x, eye, w_k, w_v


def test_one_token_each_query_head_reads_its_group_value():
    x, eye, w_k, w_v = _one_token_case()
    out = grouped_query_attention(x, eye, w_k, w_v, eye, n_heads=4, n_kv_heads=2)
    expected = torch.tensor([3.0, 3.0, 4.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        out[0, 0],
        expected,
        atol=1e-9,
        rtol=1e-9,
        msg="head mapping wrong: query head h must read key/value head h // group_size",
    )


def test_n_kv_heads_that_does_not_divide_n_heads_raises_value_error():
    x, eye, w_k, w_v = _one_token_case()
    with pytest.raises(ValueError):
        grouped_query_attention(x, eye, w_k, w_v, eye, n_heads=4, n_kv_heads=3)


def test_mask_true_means_allowed_so_a_masked_row_copies_its_own_value():
    # Two tokens and two key/value heads (n_kv_heads == n_heads, so this is ordinary MHA).
    # Token 0 may attend only to itself.
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, True]])
    out = grouped_query_attention(x, eye, eye, eye, eye, n_heads=2, n_kv_heads=2, mask=mask)
    torch.testing.assert_close(
        out[0, 0],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        atol=1e-9,
        rtol=0,
        msg="a False entry must get zero weight: check that True means allowed",
    )


def _two_token_reference(x):
    # Plain-Python reference with d_head = 1 and scale 1. Query head h reads key and value head
    # g = h // 2 (the contiguous grouping). Keys are x[:, g] and values are x[:, 2 + g].
    rows = []
    for t in range(2):
        row = []
        for h in range(4):
            g = h // 2
            scores = [x[t][h] * x[s][g] for s in range(2)]
            m = max(scores)
            w = [math.exp(v - m) for v in scores]
            z = sum(w)
            row.append(sum(w[s] / z * x[s][2 + g] for s in range(2)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def test_two_tokens_keys_and_values_are_grouped_together():
    # With two tokens the softmax has two keys, so a wrong key mapping changes the weights.
    # The one-token tests above cannot see it, because one key always gets weight 1.
    x = torch.tensor([[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 3.0]]], dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.float64)
    w_k = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=torch.float64)
    w_v = torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]], dtype=torch.float64)
    out = grouped_query_attention(x, eye, w_k, w_v, eye, n_heads=4, n_kv_heads=2)
    torch.testing.assert_close(
        out[0],
        _two_token_reference(x[0].tolist()),
        atol=1e-9,
        rtol=1e-9,
        msg=(
            "key/value grouping mistake: query head h must read key head h // group_size "
            "and value head h // group_size; a modulo mapping h % n_kv_heads on the keys "
            "pairs them wrongly, which only shows with two or more tokens"
        ),
    )
