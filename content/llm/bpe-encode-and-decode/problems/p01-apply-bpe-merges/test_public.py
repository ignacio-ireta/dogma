"""Public tests: the examples from problem.md, the rank order and the decode rule.

The hidden tests check the rank order in more cases, the overlap rule and the round trip.
"""

from __future__ import annotations

import pytest
from submission import BPETokenizer


def test_lowest_rank_is_applied_before_leftmost_pair() -> None:
    tok = BPETokenizer([("b", "c"), ("a", "b")])
    assert tok.encode("abc") == ["a", "bc", "</w>"], (
        "the merge (b, c) has rank 0, so it must be applied before (a, b), which has rank 1; "
        "merging the leftmost pair first is the mistake here"
    )


def test_overlapping_pair_is_merged_left_to_right() -> None:
    assert BPETokenizer([("a", "a")]).encode("aaa") == ["aa", "a", "</w>"], (
        "in 'aaa' the left two a's merge first; a right-to-left scan would give ['a', 'aa', '</w>']"
    )


def test_decode_inverts_encode_and_checks_the_marker() -> None:
    tok = BPETokenizer([("b", "c"), ("a", "b")])
    assert tok.decode(["a", "bc", "</w>"]) == "abc", (
        "decode must concatenate the tokens and remove the trailing end-of-word marker"
    )
    try:
        tok.decode(["a", "bc"])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a missing marker must raise ValueError, instead it raised {type(exc).__name__}"
        )
    pytest.fail("decode accepted tokens without the end-of-word marker: check the trailing marker")
