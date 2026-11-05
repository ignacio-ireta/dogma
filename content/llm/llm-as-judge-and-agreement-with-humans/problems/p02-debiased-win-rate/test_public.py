# Public tests: the examples from problem.md plus at least one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import debiased_win_rate


def test_mixed_consistent_inconsistent_and_tie_items():
    got = debiased_win_rate([("A", "B"), ("B", "A"), ("A", "A"), ("tie", "tie")])
    assert got == pytest.approx(0.5, abs=1e-12), (
        "mixed items: X-wins-both is 1, Y-wins-both is 0, and the rest are 0.5"
    )


def test_pure_position_bias_gives_one_half():
    got = debiased_win_rate([("A", "A"), ("A", "A"), ("A", "A")])
    assert got == pytest.approx(0.5, abs=1e-12), (
        "always choosing position A is position bias, not a preference: it must give 0.5"
    )


def test_empty_judgments_are_rejected():
    with pytest.raises(ValueError):
        debiased_win_rate([])
