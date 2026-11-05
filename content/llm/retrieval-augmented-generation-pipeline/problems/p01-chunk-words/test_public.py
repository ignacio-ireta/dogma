# Public tests: the examples from problem.md plus at least one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import chunk_words


def test_ten_words_size_four_overlap_two_gives_four_windows():
    got = chunk_words("a b c d e f g h i j", 4, 2)
    expected = ["a b c d", "c d e f", "e f g h", "g h i j"]
    if len(got) != len(expected):
        pytest.fail(
            "wrong number of windows: a missing window or a trailing window that repeats the "
            "previous tail; check the stop rule and the step",
            pytrace=False,
        )
    assert got == expected, "window contents wrong: check the step (size - overlap)"


def test_last_window_stops_the_loop_without_a_contained_tail():
    got = chunk_words("a b c d e", 4, 2)
    if len(got) != 2:
        pytest.fail(
            "extra trailing window: stop after the first window that reaches the last word",
            pytrace=False,
        )
    assert got == ["a b c d", "c d e"], "window contents wrong: check the step (size - overlap)"


def test_whitespace_is_collapsed_and_one_window_can_cover_everything():
    assert chunk_words("  one   two\nthree ", 5, 1) == ["one two three"], (
        "word boundaries must come from whitespace runs, not single spaces"
    )


def test_empty_text_gives_no_chunks_not_an_empty_string():
    assert chunk_words("", 3, 1) == [], "empty text must give [], not [''] or an error"


def test_overlap_equal_to_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_words("a b c", 3, 3)
