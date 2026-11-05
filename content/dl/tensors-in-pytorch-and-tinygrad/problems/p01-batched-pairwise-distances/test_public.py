from __future__ import annotations

import torch
from submission import batched_sq_distances


def test_single_batch_matches_hand_computed_distances():
    X = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    Y = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    out = batched_sq_distances(X, Y)
    expected = torch.tensor([[[0.0, 1.0], [25.0, 20.0]]])
    assert out.shape == (1, 2, 2), "output shape must be (B, N, M)"
    assert out.dtype == X.dtype, "result dtype must follow the float32 inputs"
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5), "wrong pairwise squared distances"


def test_batch_elements_are_independent_with_one_feature():
    X = torch.tensor([[[1.0], [2.0]], [[0.0], [0.0]]])
    Y = torch.tensor([[[0.0]], [[5.0]]])
    out = batched_sq_distances(X, Y)
    expected = torch.tensor([[[1.0], [4.0]], [[25.0], [25.0]]])
    assert out.dtype == X.dtype, "result dtype must follow the float32 inputs"
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5), (
        "batch elements must not mix their points"
    )


def test_empty_query_set_gives_empty_result_with_batch_and_reference_sizes():
    X = torch.zeros(3, 0, 2)
    Y = torch.ones(3, 4, 2)
    out = batched_sq_distances(X, Y)
    assert out.shape == (3, 0, 4), "empty query dimension must survive as (B, 0, M)"
