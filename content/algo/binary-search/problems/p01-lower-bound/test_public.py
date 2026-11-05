"""Public tests: the problem.md examples, the duplicate run, and the outside-the-range cases."""

from submission import lower_bound  # type: ignore[import-not-found]


def test_example_returns_the_first_index_of_a_run_of_duplicates() -> None:
    assert lower_bound([1, 2, 2, 2, 5], 2) == 1, (
        "duplicates: return the first index of the run of equal values, not the last"
    )


def test_target_between_two_elements_returns_the_index_of_the_next_larger_one() -> None:
    assert lower_bound([1, 3, 5], 4) == 2, (
        "target between two elements: return the index of the first element that is larger"
    )


def test_target_outside_the_range_and_an_empty_list() -> None:
    assert lower_bound([1, 3, 5], 0) == 0, "target below every element: the answer is index 0"
    assert lower_bound([1, 3, 5], 9) == 3, (
        "target above every element: the answer is len(sorted_values), not len - 1"
    )
    assert lower_bound([], 7) == 0, "empty list: the answer is 0"
