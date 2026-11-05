"""Public tests: the examples from problem.md, empty input, and the substring trap."""

from __future__ import annotations

from typing import Any

import pytest
from submission import longest_common_subsequence


def _answer(a: Any, b: Any) -> Any:
    try:
        return longest_common_subsequence(a, b)
    except RecursionError:
        pytest.fail("longest_common_subsequence raised RecursionError: fill a table in loops")
    except Exception as exc:
        pytest.fail(
            f"longest_common_subsequence raised {type(exc).__name__} on a valid input: {exc}"
        )


def _check(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _is_subsequence(part: str, whole: str) -> bool:
    remaining = iter(whole)
    return all(character in remaining for character in part)


def test_example_length_and_a_valid_witness() -> None:
    length, witness = _answer("ABCBDAB", "BDCABA")
    _check(
        length == 4,
        "length for ABCBDAB and BDCABA must be 4: when the last characters differ, take "
        "the larger of the two neighbouring entries",
    )
    _check(
        len(witness) == length,
        "the witness must have exactly the returned length: rebuild it by walking back the table",
    )
    _check(
        _is_subsequence(witness, "ABCBDAB") and _is_subsequence(witness, "BDCABA"),
        "the witness must be a subsequence of both strings, in order",
    )


def test_disjoint_alphabets_return_zero_and_the_empty_witness() -> None:
    _check(
        _answer("abc", "def") == (0, ""),
        "no shared character means the empty subsequence, length 0",
    )


def test_substring_is_not_enough_for_the_contiguous_trap() -> None:
    length, witness = _answer("abcde", "aXcXe")
    _check(
        length == 3,
        "abcde and aXcXe share the subsequence a, c, e of length 3; a contiguous block gives 1",
    )
    _check(witness == "ace", "the only longest common subsequence here is ace")
