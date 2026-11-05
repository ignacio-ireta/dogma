# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
import torch
from submission import apply_rope_scaled

F64 = torch.float64


def _call(q, k, offset, scale):
    """Call the learner; a crash on valid input is reported with the likely cause."""
    try:
        return apply_rope_scaled(q, k, offset, scale)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"valid input crashed ({type(exc).__name__}): check the (B, H, T, d) indexing and "
            "that the scaled positions broadcast over the batch and head axes"
        )


def test_scale_one_on_position_zero_is_unchanged():
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=F64)
    q_rot, _ = _call(q, torch.zeros_like(q), 0, 1)
    assert torch.allclose(q_rot, q), "scale 1 at position 0 must be the identity"


def test_scale_two_halves_the_rotation_of_the_next_token():
    q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=F64)
    q_rot, _ = _call(q, torch.zeros_like(q), 0, 2)
    expected = torch.tensor([[[[1.0, 0.0], [0.8775826, 0.4794255]]]], dtype=F64)
    assert torch.allclose(q_rot, expected, atol=1e-6), (
        "interpolation is wrong: token 1 with scale 2 sits at position 1/2, angle 0.5 radians"
    )


def test_offset_is_divided_by_the_scale_together_with_the_position():
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=F64)
    q_rot, _ = _call(q, torch.zeros_like(q), 3, 2)
    expected = torch.tensor([[[[0.0707372, 0.9974950]]]], dtype=F64)
    assert torch.allclose(q_rot, expected, atol=1e-6), (
        "offset handling is wrong: the position is (offset + t) / scale, so offset 3 at scale 2 "
        "gives 1.5 radians"
    )


def test_zero_scale_raises_value_error():
    q = torch.ones(1, 1, 2, 2, dtype=F64)
    try:
        apply_rope_scaled(q, q, 0, 0)
    except ValueError:
        return
    pytest.fail("scale 0 was accepted: a scale must be a finite number greater than 0")
