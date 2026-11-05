# Public tests: the examples from problem.md plus at least one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import cohens_kappa


def test_two_of_four_agree_on_yes_gives_half():
    got = cohens_kappa(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"], ["yes", "no"])
    assert got == pytest.approx(0.5, abs=1e-12), (
        "wrong chance correction: p_e is the sum of products of the two raters' marginals"
    )


def test_perfect_agreement_over_a_larger_label_set_is_one():
    got = cohens_kappa(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c", "d"])
    assert got == pytest.approx(1.0, abs=1e-12), (
        "perfect agreement must give kappa 1 even when a label is never used"
    )


def test_one_label_for_every_item_is_undefined_not_nan():
    try:
        cohens_kappa(["a", "a", "a"], ["a", "a", "a"], ["a", "b"])
    except ValueError as err:
        assert "undefined" in str(err), (
            "p_e == 1 raises ValueError, but must say kappa is undefined"
        )
    else:
        raise AssertionError("p_e == 1 must raise ValueError, not return a number or nan")


def test_label_outside_the_label_set_is_rejected():
    with pytest.raises(ValueError):
        cohens_kappa(["a", "z"], ["a", "a"], ["a", "b"])
