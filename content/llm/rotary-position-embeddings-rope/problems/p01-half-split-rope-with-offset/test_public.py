# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import apply_rope_qk

F64 = torch.float64


def _call(q, k, offset):
    """Call the learner; a crash on valid input is reported with the likely cause."""
    try:
        return apply_rope_qk(q, k, offset)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"valid input crashed ({type(exc).__name__}): check the (B, H, T, d) indexing and "
            "that the angles broadcast over the batch and head axes"
        )


def test_single_token_at_position_zero_is_unchanged():
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=F64)
    q_rot, _ = _call(q, torch.zeros_like(q), 0)
    assert torch.allclose(q_rot, q), (
        "position 0 must be the identity: check the pairing (i with i + d/2) and the angle at t = 0"
    )


def test_two_wide_example_rotates_by_one_radian_at_position_one():
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=F64)
    q_rot, _ = _call(q, torch.zeros_like(q), 1)
    expected = torch.tensor([[[[0.5403023, 0.8414710]]]], dtype=F64)
    assert torch.allclose(q_rot, expected, atol=1e-6), (
        "rotation at position 1 is wrong: the angle is (offset + t) * theta_i with theta_0 = 1"
    )


def test_half_split_pairs_feature_i_with_feature_i_plus_half():
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=F64)
    q_rot, _ = _call(q, torch.zeros_like(q), 1)
    expected = torch.tensor([[[[0.5403023, 0.0, 0.8414710, 0.0]]]], dtype=F64)
    assert torch.allclose(q_rot, expected, atol=1e-6), (
        "pairs are wrong: half-split pairs feature i with feature i + d/2, not with i + 1"
    )


def test_odd_feature_width_raises_value_error():
    x = torch.zeros(1, 1, 1, 3, dtype=F64)
    try:
        apply_rope_qk(x, x, 0)
    except ValueError:
        return
    pytest.fail("odd width 3 was accepted: an odd d has no complete pair, so raise ValueError")
