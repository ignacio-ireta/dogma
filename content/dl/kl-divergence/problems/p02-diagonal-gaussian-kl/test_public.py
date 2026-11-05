# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import diagonal_gaussian_kl


def test_identical_gaussians_have_zero_divergence():
    assert diagonal_gaussian_kl([0.0], [1.0], [0.0], [1.0]) == pytest.approx(0.0, abs=1e-12), (
        "identical distributions must have KL exactly 0: check the -1/2 constant and the sign "
        "of the log-scale term"
    )


def test_unit_variance_mean_shift_gives_one_half():
    assert diagonal_gaussian_kl([0.0], [1.0], [1.0], [1.0]) == pytest.approx(0.5, abs=1e-12), (
        "a mean shift of 1 at unit variance must give (0 - 1)^2 / 2: check the mean term, its "
        "1/2 factor, and that the result is in nats, not bits"
    )


def test_coordinates_add_and_variance_is_not_standard_deviation():
    value = diagonal_gaussian_kl([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 2.0])
    assert value == pytest.approx(0.8181471805599453, abs=1e-12), (
        "the coordinates must add, and the spread term must use the variance sigma^2 of Q in the "
        "denominator: check the direction, since KL(P || Q) is not KL(Q || P)"
    )


def test_non_positive_standard_deviation_is_rejected():
    with pytest.raises(ValueError):
        diagonal_gaussian_kl([0.0], [0.0], [0.0], [1.0])
