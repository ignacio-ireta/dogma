# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import torch
from submission import symmetric_infonce_loss

I2 = torch.eye(2, dtype=torch.float64)


def test_orthonormal_pairs_at_unit_temperature_match_hand_value():
    """S = I: each direction gives log(1 + e^-1), and the two directions agree."""
    loss = symmetric_infonce_loss(I2, I2, temperature=1.0)
    want = math.log(1 + math.exp(-1))
    assert abs(float(loss) - want) < 1e-9, "orthonormal pairs at tau = 1 must give log(1 + e^-1)"


def test_small_temperature_sharpens_the_loss_towards_zero():
    """Edge case: tau = 0.1 scales the diagonal to 10, so each row loss is log(1 + e^-10)."""
    loss = symmetric_infonce_loss(I2, I2, temperature=0.1)
    want = math.log(1 + math.exp(-10))
    assert abs(float(loss) - want) < 1e-9, "a smaller temperature must sharpen the softmax"


def test_row_lengths_do_not_change_the_loss():
    """The embeddings are normalised inside the loss, so scaling a row leaves it unchanged."""
    a = torch.tensor([[2.0, 0.0], [0.0, 5.0]], dtype=torch.float64)
    loss = symmetric_infonce_loss(a, I2, temperature=1.0)
    want = math.log(1 + math.exp(-1))
    assert abs(float(loss) - want) < 1e-9, (
        "the loss must depend on directions, not on vector lengths"
    )


def test_single_pair_has_zero_loss():
    """Edge case: with N = 1 there is one candidate, so the softmax is trivially 1."""
    loss = symmetric_infonce_loss(
        torch.tensor([[3.0, 4.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        temperature=0.5,
    )
    assert float(loss) == 0.0, "a single pair has one candidate, so the loss must be exactly 0"
