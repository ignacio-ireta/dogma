# Public tests: the examples from problem.md plus at least one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import majority_vote


def test_clear_majority_wins():
    assert majority_vote(["42", "42", "41"]) == "42", "the answer with the most votes must win"


def test_normalisation_merges_case_and_whitespace_before_counting():
    assert majority_vote(["  Paris ", "London", "paris"]) == "paris", (
        "answers must be compared after normalising case and whitespace"
    )


def test_tie_goes_to_the_answer_that_appeared_first():
    assert majority_vote(["Paris", "London", "paris", "london"]) == "paris", (
        "tie rule: the earliest first occurrence wins, not alphabetical or last occurrence"
    )


def test_blank_answers_are_dropped_and_do_not_vote():
    assert majority_vote(["  ", "7", "7 ", ""]) == "7", (
        "blank answers must be dropped before voting; they must not win as the empty string"
    )


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        majority_vote([])
