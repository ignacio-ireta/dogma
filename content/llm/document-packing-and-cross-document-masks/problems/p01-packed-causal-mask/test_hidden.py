# Hidden tests: fixture, oracles, properties and the input contract (CONTENT_SPEC §6.4).
# Failure messages name the kind of mistake. They never print hidden inputs, shapes or values.
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import packed_causal_mask, restart_positions


def _close(actual, desired, *, err_msg, **tolerance):
    """assert_allclose whose failure shows err_msg as the first line, without the values."""
    try:
        np.testing.assert_allclose(actual, desired, err_msg=err_msg, **tolerance)
    except AssertionError:
        pytest.fail(err_msg, pytrace=False)


def _equal(actual, desired, *, err_msg):
    """assert_array_equal whose failure shows err_msg as the first line, without the values."""
    try:
        np.testing.assert_array_equal(actual, desired, err_msg=err_msg)
    except AssertionError:
        pytest.fail(err_msg, pytrace=False)


def _causal(size: int) -> np.ndarray:
    """Lower-triangular reference (True on and below the diagonal), built independently."""
    return np.tril(np.ones((size, size), dtype=bool))


def _mask_from_key_sets(key_sets: list[set[int]]) -> np.ndarray:
    out = np.zeros((len(key_sets), len(key_sets)), dtype=bool)
    for i, keys in enumerate(key_sets):
        out[i, sorted(keys)] = True
    return out


def _run_lengths(doc_ids: np.ndarray) -> list[int]:
    """Lengths of the consecutive runs of equal ids, from a plain loop over the values."""
    lengths: list[int] = []
    values = doc_ids.tolist()
    for idx, value in enumerate(values):
        if idx > 0 and value == values[idx - 1]:
            lengths[-1] += 1
        else:
            lengths.append(1)
    return lengths


def _block_diagonal_oracle(doc_ids: np.ndarray) -> np.ndarray:
    """Reference: one lower-triangular block per document, placed on the diagonal."""
    size = doc_ids.shape[0]
    out = np.zeros((size, size), dtype=bool)
    start = 0
    for length in _run_lengths(doc_ids):
        out[start : start + length, start : start + length] = _causal(length)
        start += length
    return out


def _positions_oracle(doc_ids: np.ndarray) -> np.ndarray:
    """Reference: 0, 1, ..., length-1 for each run, concatenated."""
    return np.concatenate([np.arange(n) for n in _run_lengths(doc_ids)]).astype(np.int64)


def _expect_value_error(call, arg, mistake: str) -> None:
    try:
        call(arg)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def _mask(doc_ids: np.ndarray) -> np.ndarray:
    try:
        mask = packed_causal_mask(doc_ids)
    except Exception as exc:
        pytest.fail(f"packed_causal_mask raised {type(exc).__name__} on a valid row")
    assert isinstance(mask, np.ndarray), "return a NumPy array, not a list"
    size = doc_ids.shape[0]
    assert mask.shape == (size, size), (
        "mask shape mismatch: the mask needs one row and one column per token"
    )
    assert mask.dtype == np.bool_, "mask must be a boolean array (True means may attend)"
    return mask


def _check_rules(mask: np.ndarray, doc_ids: np.ndarray) -> None:
    """Named checks, so each failing rule reports the kind of mistake it catches."""
    size = doc_ids.shape[0]
    same_document = doc_ids[:, None] == doc_ids[None, :]
    assert not (mask & ~same_document).any(), (
        "cross-document leak: a query attends a key from another document"
    )
    assert not (mask & ~_causal(size)).any(), "future leak: a query attends a later key"
    assert np.all(np.diagonal(mask)), "a token must be able to attend itself: diagonal is not True"
    assert not (same_document & _causal(size) & ~mask).any(), (
        "missing key: a query cannot see an earlier token of its own document"
    )


@st.composite
def packed_rows(draw, max_docs: int = 6, max_len: int = 6):
    count = draw(st.integers(min_value=1, max_value=max_docs))
    lengths = draw(st.lists(st.integers(1, max_len), min_size=count, max_size=count))
    labels = draw(st.lists(st.integers(-1000, 1000), min_size=count, max_size=count, unique=True))
    pairs = zip(labels, lengths, strict=True)
    return np.array([lab for lab, n in pairs for _ in range(n)], dtype=np.int64)


def test_hand_computed_eight_token_mask_matches_block_structure():
    # Ids 5, 9 and 2 label three documents of lengths 2, 4 and 2. No id reappears.
    doc_ids = np.array([5, 5, 9, 9, 9, 9, 2, 2])
    expected = _mask_from_key_sets([{0}, {0, 1}, {2}, {2, 3}, {2, 3, 4}, {2, 3, 4, 5}, {6}, {6, 7}])
    mask = _mask(doc_ids)
    _check_rules(mask, doc_ids)
    _equal(mask, expected, err_msg="packed mask differs from the hand-computed block structure")


def test_hand_computed_restart_positions_for_eight_tokens():
    _equal(
        restart_positions([5, 5, 9, 9, 9, 9, 2, 2]),
        [0, 1, 0, 1, 2, 3, 0, 1],
        err_msg="position ids must restart at 0 at the first token of each document",
    )


@pytest.mark.parametrize("label", [0, -7, 42], ids=["zero", "negative", "large"])
def test_single_document_row_equals_plain_causal_mask(label):
    for size in range(1, 11):
        doc_ids = np.full(size, label, dtype=np.int64)
        _equal(
            _mask(doc_ids),
            _causal(size),
            err_msg="one document must give the plain causal mask: the causal rule is wrong",
        )


def test_mask_matches_block_diagonal_oracle_on_seeded_rows():
    rng = np.random.default_rng(7)
    for _ in range(50):
        count = int(rng.integers(1, 6))
        lengths = rng.integers(1, 7, size=count)
        labels = rng.permutation(100)[:count]
        doc_ids = np.repeat(labels, lengths).astype(np.int64)
        _equal(
            _mask(doc_ids),
            _block_diagonal_oracle(doc_ids),
            err_msg="mask differs from the block-diagonal causal reference",
        )


def test_restart_positions_match_per_document_ranges_on_seeded_rows():
    rng = np.random.default_rng(13)
    for _ in range(50):
        count = int(rng.integers(1, 6))
        lengths = rng.integers(1, 7, size=count)
        labels = rng.permutation(100)[:count]
        doc_ids = np.repeat(labels, lengths).astype(np.int64)
        _equal(
            restart_positions(doc_ids),
            _positions_oracle(doc_ids),
            err_msg="position ids differ from the per-document ranges",
        )


@settings(report_multiple_bugs=False, max_examples=80, deadline=None)
@given(packed_rows())
def test_no_key_from_another_document_is_ever_visible(doc_ids):
    _check_rules(_mask(doc_ids), doc_ids)


@settings(report_multiple_bugs=False, max_examples=80, deadline=None)
@given(packed_rows())
def test_row_allows_exactly_its_position_in_document_plus_one(doc_ids):
    counts = _mask(doc_ids).sum(axis=1)
    assert np.array_equal(counts, _positions_oracle(doc_ids) + 1), (
        "a query must attend every earlier token of its document and no other key: "
        "the row count is wrong"
    )


@settings(report_multiple_bugs=False, max_examples=60, deadline=None)
@given(packed_rows())
def test_every_token_attends_itself(doc_ids):
    assert np.all(np.diagonal(_mask(doc_ids))), "a token must be able to attend itself"


@settings(report_multiple_bugs=False, max_examples=60, deadline=None)
@given(packed_rows())
def test_restart_positions_reset_at_each_document_start(doc_ids):
    pos = restart_positions(doc_ids)
    starts = np.r_[True, doc_ids[1:] != doc_ids[:-1]]
    assert np.all(pos[starts] == 0), "position ids must restart at 0 at each document start"
    inside = ~starts[1:]
    assert np.all(pos[1:][inside] == pos[:-1][inside] + 1), (
        "position ids must count up by one inside each document"
    )


@pytest.mark.parametrize("doc_ids", [[0, 1, 0], [4, 4, 2, 4]], ids=["id-returns", "ids-interleave"])
def test_non_contiguous_document_ids_raise_value_error(doc_ids):
    mistake = "an id that reappears after another document must raise ValueError"
    _expect_value_error(packed_causal_mask, doc_ids, mistake)
    _expect_value_error(restart_positions, doc_ids, mistake)


@pytest.mark.parametrize(
    "doc_ids",
    [[], np.zeros(0, dtype=np.int64), [[0, 0], [1, 1]], np.zeros((2, 2), dtype=int)],
    ids=["empty-list", "empty-int-array", "list-of-lists", "two-dimensional-array"],
)
def test_empty_or_two_dimensional_ids_raise_value_error(doc_ids):
    mistake = "doc_ids must be a non-empty 1-D array: raise ValueError for other shapes"
    _expect_value_error(packed_causal_mask, doc_ids, mistake)
    _expect_value_error(restart_positions, doc_ids, mistake)


@pytest.mark.parametrize("doc_ids", [[0.0, 0.0, 1.0], [True, False]], ids=["float-ids", "bool-ids"])
def test_non_integer_document_ids_raise_value_error(doc_ids):
    _expect_value_error(
        packed_causal_mask,
        doc_ids,
        "doc_ids must be integers: raise ValueError for float or bool input",
    )


def test_inputs_are_not_mutated():
    doc_ids = np.array([2, 2, 7, 7, 7], dtype=np.int64)
    before = doc_ids.copy()
    packed_causal_mask(doc_ids)
    restart_positions(doc_ids)
    _equal(doc_ids, before, err_msg="doc_ids was modified in place")


def test_outputs_have_documented_dtype_and_shape():
    mask = packed_causal_mask([3, 3, 8])
    pos = restart_positions([3, 3, 8])
    assert mask.shape == (3, 3) and mask.dtype == np.bool_, "mask must be (T, T) bool"
    assert pos.shape == (3,) and pos.dtype == np.int64, "positions must be (T,) int64"


def test_scalar_document_id_raises_value_error():
    mistake = "a scalar doc_ids must raise ValueError: doc_ids must be 1-D"
    _expect_value_error(packed_causal_mask, 3, mistake)
    _expect_value_error(restart_positions, 3, mistake)
