from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import majority_vote

from dogma_data import load_frame


def _raises(expected, call, mistake):
    try:
        call()
    except expected:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


POOL = ["Yes", "no", " 7 ", "7", "Paris", "paris ", "a  b", "a b", "", "  "]


def checked(call):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input raised: honor the documented contract", pytrace=False)
    if not isinstance(result, str):
        pytest.fail("return the normalised winner as a str", pytrace=False)
    return result


def rejects(call, exc):
    try:
        call()
    except exc:
        return
    except Exception:
        pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)
    pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)


def reference(answers: list[str]) -> str:
    # Independent path: a list of normalised survivors, then counts in first-occurrence order.
    survivors = [" ".join(a.split()).lower() for a in answers]
    survivors = [a for a in survivors if a]
    if not survivors:
        raise ValueError("no survivors")
    order: list[str] = []
    tally: dict[str, int] = {}
    for a in survivors:
        if a not in tally:
            order.append(a)
            tally[a] = 0
        tally[a] += 1
    best = max(tally.values())
    return next(a for a in order if tally[a] == best)


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.lists(st.sampled_from(POOL), min_size=1, max_size=25))
def test_random_answer_lists_match_the_first_occurrence_oracle(answers):
    if all(not " ".join(a.split()) for a in answers):
        _raises(
            ValueError,
            lambda: majority_vote(answers),
            "answers with no non-blank text must raise ValueError",
        )
        return
    got = checked(lambda: majority_vote(answers))
    want = reference(answers)
    if got != want:
        pytest.fail(
            "wrong winner: check normalisation (collapse whitespace, lowercase) and the tie rule "
            "(earliest first occurrence)",
            pytrace=False,
        )


def test_tie_at_the_start_of_the_list_goes_to_the_first_answer():
    assert checked(lambda: majority_vote(["b", "a", "a", "b"])) == "b", (
        "2-2 tie: the answer that appeared first in the list must win"
    )


def test_later_first_occurrence_loses_a_tie_even_with_more_appearances_later():
    got = checked(lambda: majority_vote(["x", "y", "y", "x", "z", "z", "y"]))
    assert got == "y", "y has 3 votes and beats x and z: the count decides before the tie rule"


def test_three_way_tie_is_settled_by_first_occurrence():
    assert checked(lambda: majority_vote(["c", "a", "b", "b", "a", "c"])) == "c", (
        "three-way 2-2-2 tie: the first answer in the list, c, must win"
    )


def test_case_and_internal_whitespace_are_normalised():
    assert (
        checked(lambda: majority_vote(["New   York", "new york", "NEW\tYORK", "boston"]))
        == "new york"
    ), "answers that differ only by case or whitespace runs must count as the same answer"


def test_punctuation_is_kept_so_trailing_dot_is_a_different_answer():
    assert checked(lambda: majority_vote(["3.", "3.", "3"])) == "3.", (
        "punctuation is not normalised: '3.' and '3' are different answers, so '3.' wins 2 to 1"
    )


@pytest.fixture(scope="module")
def gsm8k_grouped_answer() -> str:
    """Return one final answer with a thousands separator from the pinned GSM8K test snapshot."""
    for answer in load_frame("gsm8k")["answer"]:
        final = str(answer).rsplit("####", 1)[-1].strip()
        if "," in final:
            return final
    pytest.fail("the pinned GSM8K snapshot must include a comma-grouped numeric answer")


def test_real_gsm8k_thousands_separator_remains_punctuation(
    gsm8k_grouped_answer: str,
) -> None:
    ungrouped = gsm8k_grouped_answer.replace(",", "")
    got = checked(lambda: majority_vote([gsm8k_grouped_answer, ungrouped]))
    if got != gsm8k_grouped_answer:
        pytest.fail(
            "GSM8K numeric punctuation is significant: a comma-grouped answer and its ungrouped "
            "spelling remain distinct, so the first wins this tie",
            pytrace=False,
        )


def test_blank_only_input_is_rejected():
    rejects(lambda: majority_vote(["", "   ", "\n"]), ValueError)


def test_empty_list_is_rejected():
    rejects(lambda: majority_vote([]), ValueError)


def test_bare_string_is_rejected_not_read_as_characters():
    rejects(lambda: majority_vote("aab"), TypeError)  # type: ignore[arg-type]


def test_non_string_elements_are_rejected():
    rejects(lambda: majority_vote(["1", 1]), TypeError)  # type: ignore[list-item]
    rejects(lambda: majority_vote([None]), TypeError)  # type: ignore[list-item]


def test_strict_majority_does_not_depend_on_the_order_of_the_list():
    base = ["red"] * 5 + ["blue"] * 2 + ["green"] * 2
    for shift in range(len(base)):
        rotated = base[shift:] + base[:shift]
        assert checked(lambda rotated=rotated: majority_vote(rotated)) == "red", (
            "a strict majority must win wherever it sits in the list"
        )
    reversed_list = list(reversed(base))
    assert checked(lambda: majority_vote(reversed_list)) == "red", (
        "a strict majority must win in reversed order too"
    )


def test_input_list_is_not_mutated():
    answers = [" Yes ", "yes", "No"]
    before = list(answers)
    checked(lambda: majority_vote(answers))
    assert answers == before, "the input list must not be changed"


def test_large_sample_count_is_counted_correctly():
    answers = ["a"] * 4000 + ["b"] * 3999 + ["c"] * 2
    assert checked(lambda: majority_vote(answers)) == "a", (
        "a 4000 to 3999 margin must be counted exactly over a long list"
    )


def test_tuple_of_answers_is_accepted():
    # Two of the three samples say "42", so "42" wins; only the container type differs from a list.
    result = checked(lambda: majority_vote(("42", "42", "41")))
    assert result == "42", (
        "a tuple of answers is a valid input: it must be voted on the same way as a list"
    )
