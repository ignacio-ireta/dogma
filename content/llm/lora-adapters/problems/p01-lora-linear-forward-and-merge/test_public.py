# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
from __future__ import annotations

import pytest
import torch
from submission import lora_linear


def _example():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    W = torch.eye(2, dtype=torch.float64)
    A = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    return x, W, A, B


def test_worked_example_adds_scaled_low_rank_update_to_base_output():
    x, W, A, B = _example()
    out = lora_linear(x, W, None, A, B, alpha=2.0)
    torch.testing.assert_close(
        out,
        torch.tensor([[1.0, 4.0]], dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
        msg="the adapter term must be (alpha / r) * x A^T B^T, added to x W^T",
    )


def test_merged_path_gives_the_same_output_as_the_unmerged_path():
    x, W, A, B = _example()
    unmerged = lora_linear(x, W, None, A, B, alpha=2.0, merge=False)
    merged = lora_linear(x, W, None, A, B, alpha=2.0, merge=True)
    torch.testing.assert_close(
        merged,
        unmerged,
        atol=1e-9,
        rtol=1e-9,
        msg="merged and unmerged outputs differ: the merged weight must be W + (alpha / r) B A",
    )


def test_zero_b_returns_the_base_output_so_training_starts_at_the_base_model():
    x, W, A, _ = _example()
    B0 = torch.zeros(2, 1, dtype=torch.float64)
    torch.testing.assert_close(
        lora_linear(x, W, None, A, B0, alpha=2.0),
        x @ W.T,
        atol=1e-12,
        rtol=0,
        msg="with B = 0 the output must equal the base layer: the adapter adds nothing at step 0",
    )


def test_mismatched_b_shape_raises_value_error():
    x, W, A, _ = _example()
    bad_B = torch.zeros(3, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        lora_linear(x, W, None, A, bad_B, alpha=2.0)
