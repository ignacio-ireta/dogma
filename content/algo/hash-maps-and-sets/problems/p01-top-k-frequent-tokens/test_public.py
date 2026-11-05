"""Public tests: the examples from problem.md, the alphabetical tie-break, and k edge cases."""

from submission import top_k_frequent  # type: ignore[import-not-found]


def test_example_returns_the_two_most_frequent_tokens_with_counts() -> None:
    tokens = ["the", "cat", "the", "dog", "cat", "the"]
    assert top_k_frequent(tokens, 2) == [("the", 3), ("cat", 2)], (
        "wrong top two: count each token, then rank by count, highest first"
    )


def test_equal_counts_are_ordered_alphabetically_not_by_first_appearance() -> None:
    assert top_k_frequent(["b", "a", "c", "a", "b"], 2) == [("a", 2), ("b", 2)], (
        "tie in count: order the tied tokens alphabetically (a before b), not by first appearance"
    )


def test_k_zero_and_empty_input_both_return_an_empty_list() -> None:
    assert top_k_frequent(["x", "y"], 0) == [], "k = 0 should return an empty list"
    assert top_k_frequent([], 3) == [], "no tokens should return an empty list"


def test_k_larger_than_the_vocabulary_returns_every_distinct_token_in_order() -> None:
    assert top_k_frequent(["x", "y", "x"], 10) == [("x", 2), ("y", 1)], (
        "k larger than the number of distinct tokens should return all of them, sorted"
    )
