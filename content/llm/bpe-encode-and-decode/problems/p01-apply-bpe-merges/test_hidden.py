"""Hidden tests: a naive encoder as the oracle, round trips, rank order and the validation families.

Failure messages name the kind of mistake and carry no hidden merge list, word or answer.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, NoReturn

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import BPETokenizer

END = "</w>"


def _fail(why: str) -> NoReturn:
    """The one place these tests raise, so every mistake is reported with a single named line."""
    raise AssertionError(why)


def _expect_error(exc_type: type[BaseException], make: Callable[[], Any], message: str) -> None:
    """Pass when make() raises exc_type. Otherwise fail with message, naming what happened."""
    try:
        make()
    except exc_type:
        return
    except Exception as exc:
        wrong = type(exc).__name__
    else:
        _fail(f"{message}: nothing was raised")
    _fail(f"{message}: got {wrong} instead")


def _naive_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Independent oracle. It finds the best pair with a plain linear scan, every time."""
    rank = {pair: index for index, pair in enumerate(merges)}
    seq = [*word, END]
    while True:
        best_rank = None
        best_pair: tuple[str, str] | None = None
        for pos in range(len(seq) - 1):
            key = (seq[pos], seq[pos + 1])
            if key in rank and (best_rank is None or rank[key] < best_rank):
                best_rank, best_pair = rank[key], key
        if best_pair is None:
            return seq
        first, second = best_pair
        nxt: list[str] = []
        pos = 0
        while pos < len(seq):
            if pos + 1 < len(seq) and seq[pos] == first and seq[pos + 1] == second:
                nxt.append(first + second)
                pos += 2
            else:
                nxt.append(seq[pos])
                pos += 1
        seq = nxt


def _random_merges(rng: np.random.Generator) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for _ in range(int(rng.integers(0, 9))):
        left = "".join("abc"[int(rng.integers(3))] for _ in range(int(rng.integers(1, 3))))
        right = "".join("abc"[int(rng.integers(3))] for _ in range(int(rng.integers(1, 3))))
        if (left, right) not in merges:
            merges.append((left, right))
    return merges


def _encode(tok: BPETokenizer, word: str) -> list[str]:
    try:
        got = tok.encode(word)
    except Exception as exc:
        error: str | None = type(exc).__name__
    else:
        error = None
    if error is not None:
        _fail(f"your encode raised {error} on a valid word: check the merge loop")
    if type(got) is not list or any(type(s) is not str for s in got):
        _fail("encode must return a list of str, one symbol per token")
    return got


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_merge_lists_match_the_naive_encoder(seed: int) -> None:
    rng = np.random.default_rng(seed)
    merges = _random_merges(rng)
    word = "".join("abc"[int(rng.integers(3))] for _ in range(int(rng.integers(0, 9))))
    got = _encode(BPETokenizer(merges), word)
    if got != _naive_encode(word, merges):
        _fail(
            "encoded symbols differ from the naive encoder: apply the lowest rank first, merge "
            "every occurrence, and recompute the candidate pairs after each merge"
        )


def test_seeded_sweep_matches_the_naive_encoder() -> None:
    # Hypothesis draws only a few dozen seeds; this sweep covers 3000 fixed ones, so a rule that
    # is wrong in a few percent of cases (leftmost pair first, highest rank first) cannot hide.
    for seed in range(3000):
        rng = np.random.default_rng(seed)
        merges = _random_merges(rng)
        word = "".join("abc"[int(rng.integers(3))] for _ in range(int(rng.integers(0, 9))))
        got = _encode(BPETokenizer(merges), word)
        if got != _naive_encode(word, merges):
            _fail(
                "encoded symbols differ from the naive encoder on a fixed seed: apply the lowest "
                "rank first, merge every occurrence, and recompute the candidate pairs after each "
                "merge"
            )


@settings(max_examples=80, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    word=st.text(alphabet="abcé", max_size=10),
    merges=st.lists(
        st.tuples(st.text("abc", min_size=1, max_size=2), st.text("abc", min_size=1, max_size=2)),
        max_size=6,
        unique=True,
    ),
)
def test_decode_inverts_encode_for_any_merge_list(word: str, merges: list[tuple[str, str]]) -> None:
    tok = BPETokenizer(merges)
    got = tok.decode(_encode(tok, word))
    if got != word:
        _fail(
            "decode does not invert encode: concatenate the tokens and remove exactly the trailing "
            "marker"
        )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    word=st.text(alphabet="abcé", max_size=10),
    merges=st.lists(
        st.tuples(st.text("abc", min_size=1, max_size=2), st.text("abc", min_size=1, max_size=2)),
        max_size=6,
        unique=True,
    ),
)
def test_symbols_concatenate_to_the_word_and_the_marker(
    word: str, merges: list[tuple[str, str]]
) -> None:
    got = _encode(BPETokenizer(merges), word)
    if "".join(got) != word + END:
        _fail(
            "the symbols must concatenate to the word followed by the marker: a merge must join "
            "its two parts exactly, and no character may be dropped"
        )


def test_lowest_rank_wins_even_when_it_is_to_the_right() -> None:
    got = BPETokenizer([("b", "c"), ("a", "b")]).encode("bcab")
    if got != ["bc", "ab", END]:
        _fail("apply the lowest-rank pair wherever it is in the word, not the leftmost pair")


def test_a_symbol_made_by_an_earlier_merge_can_outrank_a_later_rule() -> None:
    got = BPETokenizer([("ab", "x"), ("a", "b")]).encode("abx")
    if got != ["abx", END]:
        _fail(
            "after (a, b) makes 'ab', the pair ('ab', 'x') has rank 0 and must be applied, so the "
            "loop must repeat until no merge is left"
        )


def test_every_occurrence_of_the_chosen_pair_is_merged() -> None:
    got = BPETokenizer([("a", "b")]).encode("ababab")
    if got != ["ab", "ab", "ab", END]:
        _fail("merge every occurrence of the chosen pair, not only the first one")


def test_overlapping_occurrences_are_merged_left_to_right() -> None:
    got = BPETokenizer([("a", "a")]).encode("aaaa")
    if got != ["aa", "aa", END]:
        _fail("after a match, skip both symbols: 'aaaa' gives 'aa' and 'aa', with no overlap")


def test_all_occurrences_merge_before_a_new_symbol_is_considered() -> None:
    # Merging both (a, b) occurrences first gives 'ab' 'ab'. Merging one occurrence at a time would
    # let ('ab', 'a') (rank 0) join the first 'ab' with the second a, which is not the rule.
    got = BPETokenizer([("ab", "a"), ("a", "b")]).encode("abab")
    if got != ["ab", "ab", END]:
        _fail(
            "apply the chosen pair to every occurrence in one pass, then look for the next pair; "
            "merging one occurrence at a time changes the answer"
        )


def test_characters_no_merge_touches_stay_single_symbols() -> None:
    tok = BPETokenizer([("a", "b")])
    if _encode(tok, "é") != ["é", END]:
        _fail("a character that no merge touches must stay a single symbol, not become <unk>")
    if _encode(tok, "zz") != ["z", "z", END]:
        _fail("a character outside every merge must not be dropped or merged with its neighbour")


def test_empty_word_encodes_to_the_marker_and_decodes_back() -> None:
    tok = BPETokenizer([("a", "b")])
    if _encode(tok, "") != [END]:
        _fail("the empty word must encode to the marker alone")
    if tok.decode([END]) != "":
        _fail("decoding the marker alone must give the empty word")


def test_empty_merge_list_gives_one_symbol_per_character() -> None:
    if _encode(BPETokenizer([]), "cat") != ["c", "a", "t", END]:
        _fail("with no merges every character is its own symbol, followed by the marker")


def test_encoding_is_deterministic() -> None:
    tok = BPETokenizer([("b", "c"), ("a", "b"), ("bc", "a")])
    first = _encode(tok, "abcabc")
    second = _encode(tok, "abcabc")
    if first != second:
        _fail("encode must give the same answer every time for the same word")


def test_merge_list_is_not_mutated() -> None:
    merges = [("b", "c"), ("a", "b")]
    before = copy.deepcopy(merges)
    tok = BPETokenizer(merges)
    _encode(tok, "abc")
    if merges != before:
        _fail("the merge list was changed: do not sort, pop or edit the caller's list")


def test_duplicate_merge_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: BPETokenizer([("a", "b"), ("a", "b")]),
        "a pair that appears twice has two ranks, which must raise ValueError",
    )


def test_merge_as_a_list_instead_of_a_tuple_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: BPETokenizer([["a", "b"]]),  # type: ignore[list-item]  # invalid merge shape
        "each merge must be a tuple, not a list: check type(item) is tuple",
    )


def test_merge_with_three_symbols_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: BPETokenizer([("a", "b", "c")]),  # type: ignore[list-item]  # invalid merge shape
        "a merge must have exactly two symbols",
    )


def test_empty_symbol_in_a_merge_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: BPETokenizer([("", "b")]),
        "an empty symbol in a merge must raise ValueError",
    )


def test_merges_as_a_tuple_of_pairs_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: BPETokenizer((("a", "b"),)),  # type: ignore[arg-type]  # invalid container
        "merges must be a list, so a tuple of pairs must raise TypeError",
    )


def test_non_str_word_raises_type_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        TypeError,
        lambda: tok.encode(3),  # type: ignore[arg-type]  # invalid word
        "encode of a non-str must raise TypeError",
    )


def test_whitespace_in_word_raises_value_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        ValueError,
        lambda: tok.encode("a b"),
        "a word with whitespace must raise ValueError: split the text into words first",
    )


def test_marker_text_in_word_raises_value_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        ValueError,
        lambda: tok.encode("a</w>"),
        "a word containing the marker text must raise ValueError",
    )


def test_decode_without_the_marker_raises_value_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        ValueError,
        lambda: tok.decode(["a", "b"]),
        "tokens without a trailing marker must raise ValueError",
    )


def test_decode_of_an_empty_list_raises_value_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        ValueError,
        lambda: tok.decode([]),
        "an empty token list joins to '' with no marker, so it must raise ValueError",
    )


def test_decode_with_the_marker_inside_raises_value_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        ValueError,
        lambda: tok.decode(["a</w>", "b", END]),
        "the marker may appear only at the end: a marker inside the word must raise ValueError",
    )


def test_decode_with_a_non_str_token_raises_type_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        TypeError,
        lambda: tok.decode([1, END]),  # type: ignore[list-item]  # invalid token
        "a non-str token must raise TypeError",
    )


def test_decode_with_a_string_instead_of_a_list_raises_type_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        TypeError,
        lambda: tok.decode("a</w>"),  # type: ignore[arg-type]  # invalid container
        "decode takes a list of tokens, not one string",
    )


def test_long_word_matches_the_naive_encoder() -> None:
    rng = np.random.default_rng(7)
    merges = _random_merges(rng)
    word = "".join("abc"[int(rng.integers(3))] for _ in range(400))
    got = _encode(BPETokenizer(merges), word)
    if got != _naive_encode(word, merges):
        _fail("on a long word the symbols differ from the naive encoder: check the repeated loop")


def test_merge_with_a_non_str_left_symbol_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: BPETokenizer([(1, "b")]),  # type: ignore[list-item]  # invalid symbol type
        "a merge whose left symbol is not a str must raise TypeError",
    )


def test_merge_with_a_non_str_right_symbol_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: BPETokenizer([("a", 2)]),  # type: ignore[list-item]  # invalid symbol type
        "a merge whose right symbol is not a str must raise TypeError",
    )


def test_encode_of_a_list_of_strings_raises_type_error() -> None:
    tok = BPETokenizer([("a", "b")])
    _expect_error(
        TypeError,
        lambda: tok.encode(["a", "b"]),  # type: ignore[arg-type]  # invalid word container
        "encode takes one str word, so a list of strings must raise TypeError",
    )


def test_merge_with_the_marker_as_its_right_symbol_is_applied_at_the_end_of_the_word() -> None:
    # The marker is a separate symbol, so the pair (b, </w>) is an adjacent merge like any other.
    # A time box turns a run that never finishes into a named failure instead of a hang.
    import signal  # noqa: PLC0415

    class _Stuck(BaseException):
        pass

    def _expired(signum: int, frame: object) -> None:
        raise _Stuck

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        got = BPETokenizer([("b", END)]).encode("ab")
    except _Stuck:
        _fail(
            "encode did not finish: a merge that joins a character with the end-of-word marker "
            "must be applied, including at the end of the word"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    if got != ["a", "b" + END]:
        _fail(
            "a merge that joins a character with the end-of-word marker is an adjacent pair: "
            "apply it, including when the pair sits at the end of the word"
        )
