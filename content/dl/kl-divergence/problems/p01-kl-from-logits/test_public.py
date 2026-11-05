# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import numpy as np
import pytest
from submission import kl_divergence_from_logits


def test_identical_distributions_have_zero_divergence():
    value = kl_divergence_from_logits([[0.0, 0.0]], [[0.0, 0.0]])
    assert value.shape == (1,), "one KL value per row: the output must have shape (n,)"
    assert abs(value[0]) <= 1e-12, "identical distributions must have KL exactly 0"


def test_half_half_against_one_quarter_three_quarters():
    value = kl_divergence_from_logits([[0.0, 0.0]], [[0.0, np.log(3.0)]])
    assert value[0] == pytest.approx(0.14384103622589, abs=1e-10), (
        "KL value differs: check the direction (P weights the log-ratio), the sum over classes "
        "and natural logs"
    )


def test_zero_probability_under_p_adds_nothing_but_positive_p_under_zero_q_is_infinite():
    value = kl_divergence_from_logits([[0.0, -np.inf]], [[-np.inf, 0.0]])
    assert np.isposinf(value[0]), (
        "p_j > 0 where q_j = 0 must make the divergence +inf, not a finite number"
    )


def test_non_finite_logit_is_rejected():
    with pytest.raises(ValueError):
        kl_divergence_from_logits([[0.0, np.nan]], [[0.0, 0.0]])
