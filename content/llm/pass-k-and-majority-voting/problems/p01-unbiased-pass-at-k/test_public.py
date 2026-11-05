# Public tests: the examples from problem.md plus at least one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import pass_at_k


def test_k_equal_one_gives_the_fraction_correct():
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4, abs=1e-12), (
        "k = 1 must reduce to c / n: check the ratio's factor order"
    )


def test_two_of_five_with_budget_two_matches_the_hypergeometric_value():
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7, abs=1e-12), (
        "wrong k-subset probability: 1 - C(n - c, k) / C(n, k) = 1 - 3/10"
    )


def test_fewer_failing_samples_than_k_gives_exactly_one():
    assert pass_at_k(4, 3, 2) == 1.0, (
        "with n - c < k every k-subset holds a correct sample: return exactly 1.0"
    )


def test_no_passing_sample_gives_zero():
    assert pass_at_k(5, 0, 3) == 0.0, "c = 0 must give 0.0 for every k"


def test_k_above_n_is_rejected():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 4)
