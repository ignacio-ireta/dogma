from __future__ import annotations

import itertools
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import chunk_words

WORD = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=4)


def checked(call):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input raised: honor the documented contract", pytrace=False)
    if not isinstance(result, list) or any(not isinstance(c, str) for c in result):
        pytest.fail("return a list of str chunks, not word lists or other types", pytrace=False)
    return result


def rejects(call, exc):
    try:
        call()
    except exc:
        return
    except Exception:
        pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)
    pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)


def reference_chunks(words: list[str], size: int, overlap: int) -> list[str]:
    # Independent formulation: enumerate every window start, then cut after the first start whose
    # window reaches the end. Used only as the oracle for the chunk boundaries.
    if not words:
        return []
    step = size - overlap
    starts = list(range(0, len(words), step))
    cut = next(i for i, s in enumerate(starts) if s + size >= len(words))
    return [" ".join(words[s : s + size]) for s in starts[: cut + 1]]


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.lists(WORD, min_size=0, max_size=30), st.integers(1, 10), st.data())
def test_window_boundaries_match_the_enumeration_oracle(words, size, data):
    overlap = data.draw(st.integers(0, size - 1))
    got = checked(lambda: chunk_words(" ".join(words), size, overlap))
    want = reference_chunks(words, size, overlap)
    if got != want:
        pytest.fail(
            "window count or boundaries differ: check the step (size - overlap) and the stop rule",
            pytrace=False,
        )


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.lists(WORD, min_size=1, max_size=30), st.integers(1, 10), st.data())
def test_every_word_is_covered_and_no_chunk_is_contained_in_its_predecessor(words, size, data):
    overlap = data.draw(st.integers(0, size - 1))
    chunks = checked(lambda: chunk_words(" ".join(words), size, overlap))
    if not chunks:
        pytest.fail("non-empty text must give at least one chunk", pytrace=False)
    step = size - overlap
    previous_end = -1
    for index, chunk in enumerate(chunks):
        window = chunk.split()
        start = index * step
        if len(window) > size:
            pytest.fail("a chunk exceeds size words: cut windows at size", pytrace=False)
        if words[start : start + len(window)] != window:
            pytest.fail(
                "a chunk does not match the window at its start: check the step", pytrace=False
            )
        end = start + len(window)
        if end <= previous_end:
            pytest.fail(
                "a chunk is wholly contained in its predecessor: stop after the last word",
                pytrace=False,
            )
        previous_end = end
    if previous_end != len(words):
        pytest.fail(
            "the windows do not reach the last word: words are missing at the end", pytrace=False
        )


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.lists(WORD, min_size=2, max_size=40), st.integers(2, 10))
def test_consecutive_full_windows_share_exactly_overlap_words(words, size):
    overlap = size // 2
    chunks = checked(lambda: chunk_words(" ".join(words), size, overlap))
    for previous, current in itertools.pairwise(chunks):
        before = previous.split()
        after = current.split()
        if len(before) != size:
            pytest.fail("only the last chunk may be shorter than size", pytrace=False)
        if before[-overlap:] != after[:overlap]:
            pytest.fail(
                "consecutive windows do not share the overlap: check the step", pytrace=False
            )


def test_overlap_zero_partitions_the_words_exactly():
    words = [f"w{i}" for i in range(13)]
    chunks = checked(lambda: chunk_words(" ".join(words), 4, 0))
    joined = " ".join(chunks).split()
    if joined != words:
        pytest.fail("overlap 0 must partition the words with no repeats or gaps", pytrace=False)


def test_size_larger_than_document_gives_one_chunk():
    assert checked(lambda: chunk_words("alpha beta", 50, 10)) == ["alpha beta"], (
        "a document shorter than size must be one chunk, not padded"
    )


def test_whitespace_only_text_gives_no_chunks():
    assert checked(lambda: chunk_words(" \n\t  ", 3, 1)) == [], (
        "whitespace-only text has no words and must give []"
    )


def test_invalid_types_raise_type_error():
    rejects(lambda: chunk_words(["a", "b"], 2, 0), TypeError)  # type: ignore[arg-type]
    rejects(lambda: chunk_words("a b", 2.0, 0), TypeError)  # type: ignore[arg-type]
    rejects(lambda: chunk_words("a b", True, 0), TypeError)  # bool size is not an int here
    rejects(lambda: chunk_words("a b", 2, True), TypeError)  # bool is not an int here


def test_invalid_window_sizes_raise_value_error():
    rejects(lambda: chunk_words("a b", 0, 0), ValueError)
    rejects(lambda: chunk_words("a b", -1, 0), ValueError)
    rejects(lambda: chunk_words("a b", 3, -1), ValueError)
    rejects(lambda: chunk_words("a b", 3, 4), ValueError)


def test_input_string_is_not_changed():
    text = "keep  this   text"
    chunk_words(text, 2, 1)
    assert text == "keep  this   text", "the input string must not be changed"
