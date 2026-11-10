# Hidden tests. The oracle is a plain-Python reference: build the EOS-joined stream, then slice it.
# Messages name the kind of mistake; they never show hidden inputs or expected blocks.
from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import pack_next_token_blocks

TOKEN = st.integers(min_value=0, max_value=40)


def _call(documents, block_size, eos_id):
    """Run the learner on a valid call. Failures are AssertionErrors that name the mistake."""
    snapshot = [list(doc) for doc in documents]
    try:
        out = pack_next_token_blocks(documents, block_size, eos_id)
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: check the signature and the block_size "
            "and eos_id names"
        ) from None
    if [list(doc) for doc in documents] != snapshot:
        raise AssertionError("input documents were modified; do not append EOS to caller lists")
    if not isinstance(out, tuple) or len(out) != 2 or not all(torch.is_tensor(t) for t in out):
        raise AssertionError("return a tuple of two torch tensors: inputs and targets")
    inputs, targets = out
    if inputs.dtype != torch.int64 or targets.dtype != torch.int64:
        raise AssertionError("inputs and targets must be int64: token ids are not floats")
    if tuple(inputs.shape) != tuple(targets.shape) or inputs.ndim != 2:
        raise AssertionError("inputs and targets must share a two-dimensional shape (B, T)")
    if inputs.shape[1] != block_size:
        raise AssertionError("block width must equal block_size, the number of inputs per block")
    return inputs, targets


def _reference(documents, block_size, eos_id):
    """Plain-Python oracle: EOS-joined stream, then windows of T+1 with stride T."""
    stream: list[int] = []
    for doc in documents:
        stream.extend(doc)
        stream.append(eos_id)
    n_blocks = max(0, (len(stream) - 1) // block_size)
    inputs, targets = [], []
    for b in range(n_blocks):
        start = b * block_size
        inputs.append(stream[start : start + block_size])
        targets.append(stream[start + 1 : start + block_size + 1])
    return inputs, targets


def _expect_value_error(thunk, message):
    raised = None
    try:
        thunk()
    except Exception as exc:
        raised = exc
    if raised is None:
        raise AssertionError(message)
    if not isinstance(raised, ValueError):
        raise AssertionError(f"invalid argument must raise ValueError, not {type(raised).__name__}")


def _int(gen: torch.Generator, lo: int, hi: int) -> int:
    """Seeded integer in [lo, hi]."""
    return int(torch.randint(lo, hi + 1, (1,), generator=gen))


def _random_docs(gen: torch.Generator, n_docs: int, max_len: int, vocab: int):
    docs = []
    for _ in range(n_docs):
        n = _int(gen, 0, max_len)
        docs.append(torch.randint(0, vocab, (n,), generator=gen).tolist())
    return docs


def test_random_corpora_match_plain_python_reference():
    gen = torch.Generator().manual_seed(11)
    for _ in range(60):
        docs = _random_docs(gen, _int(gen, 0, 6), 10, 30)
        block = _int(gen, 1, 9)
        eos = _int(gen, 0, 29)  # may collide with real token ids on purpose
        inputs, targets = _call(docs, block, eos)
        want_in, want_tg = _reference(docs, block, eos)
        if inputs.tolist() != want_in:
            raise AssertionError(
                "inputs differ from the reference: windows start every T tokens and the "
                "stream has an EOS after each document"
            )
        if targets.tolist() != want_tg:
            raise AssertionError(
                "targets differ from the reference: targets are the window shifted left by one"
            )


def test_targets_are_inputs_shifted_by_one_position():
    gen = torch.Generator().manual_seed(12)
    docs = _random_docs(gen, 4, 8, 50)
    inputs, targets = _call(docs, 5, 0)
    if inputs.shape[0] < 2:
        raise AssertionError("test corpus too small to check the block-to-block shift")
    if not torch.equal(targets[:, :-1], inputs[:, 1:]):
        raise AssertionError("targets must equal inputs shifted left by one inside each block")


def test_consecutive_blocks_share_one_token_at_the_boundary():
    docs = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    inputs, targets = _call(docs, 4, 0)
    # Stream: 1..9, 0 (N = 10). Blocks at 0 and 4 overlap by exactly one target and input.
    if inputs.tolist()[1][0] != targets.tolist()[0][-1]:
        raise AssertionError(
            "consecutive blocks must share one token: the last target of block b is the first "
            "input of block b + 1"
        )


def test_eos_follows_every_document_in_the_stream():
    docs = [[11, 12], [], [13]]
    eos = 99
    # With block size N - 1, one block holds the whole stream: inputs S[:-1], targets S[1:].
    stream = [11, 12, 99, 99, 13, 99]
    inputs, targets = _call(docs, len(stream) - 1, eos)
    if inputs.tolist() != [stream[:-1]]:
        raise AssertionError(
            "the stream must hold one EOS after each document, empty ones included"
        )
    if targets.tolist() != [stream[1:]]:
        raise AssertionError("targets of a whole-stream block must be the stream shifted by one")


def test_remainder_is_dropped_and_no_padded_block_is_returned():
    docs = [[1, 2, 3, 4]]  # stream (1, 2, 3, 4, 0) has N = 5 tokens
    inputs, targets = _call(docs, 2, 0)
    if inputs.shape[0] != 2:
        raise AssertionError(
            "block count must be (N - 1) // T; a leftover tail is dropped, not padded"
        )
    if targets.tolist()[-1] != [4, 0]:
        raise AssertionError("the last full block must end at the EOS that fits in its window")


def test_block_larger_than_stream_returns_zero_blocks_with_width():
    inputs, targets = _call([[1]], 10, 0)  # stream (1, 0), N = 2 < T + 1
    if tuple(inputs.shape) != (0, 10) or tuple(targets.shape) != (0, 10):
        raise AssertionError("a stream too short for one window must return shape (0, block_size)")


def test_empty_documents_contribute_only_their_eos():
    inputs, targets = _call([[], []], 1, 9)  # stream (9, 9), N = 2
    if inputs.tolist() != [[9]] or targets.tolist() != [[9]]:
        raise AssertionError("an empty document must add only its EOS separator to the stream")


def test_token_ids_above_float32_precision_survive_exactly():
    big = 2**40 + 3  # above 2**24, so a float32 cast would change it
    inputs, targets = _call([[big, 7]], 2, 0)  # stream (big, 7, 0)
    if inputs.tolist() != [[big, 7]] or targets.tolist() != [[7, 0]]:
        raise AssertionError("token ids changed value: keep the stream in an int64 tensor")


def test_eos_id_may_equal_a_token_that_occurs_in_documents():
    # The EOS is a separator by position, not by value: a real token equal to eos_id is a token.
    inputs, targets = _call([[0, 0]], 2, 0)  # stream (0, 0, 0), N = 3
    if inputs.tolist() != [[0, 0]] or targets.tolist() != [[0, 0]]:
        raise AssertionError("tokens equal to eos_id must still be packed as ordinary tokens")


def test_invalid_block_size_raises_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks([[1]], 0, 0), "block_size must be at least 1"
    )
    _expect_value_error(
        lambda: pack_next_token_blocks([[1]], True, 0), "a bool block_size must raise ValueError"
    )


def test_negative_eos_id_raises_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks([[1]], 2, -1), "a negative eos_id must raise ValueError"
    )


def test_negative_token_id_in_a_document_raises_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks([[1, -4]], 2, 0), "a negative token id must raise ValueError"
    )


def test_flat_list_of_ints_instead_of_documents_raises_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks([1, 2, 3], 2, 0),  # type: ignore[arg-type]
        "documents must be a list of token-id lists; a flat list of ints is invalid",
    )


def test_inputs_are_not_mutated_when_eos_is_added():
    docs = [[1, 2], [3]]
    pack_next_token_blocks(docs, 2, 0)
    if docs != [[1, 2], [3]]:
        raise AssertionError("documents were modified; build the EOS-joined stream in a new list")


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    documents=st.lists(st.lists(TOKEN, max_size=6), max_size=5),
    block_size=st.integers(min_value=1, max_value=7),
    eos_id=TOKEN,
)
def test_property_stream_is_recoverable_from_blocks(documents, block_size, eos_id):
    inputs, targets = _call(documents, block_size, eos_id)
    stream: list[int] = []
    for doc in documents:
        stream.extend(doc)
        stream.append(eos_id)
    n_blocks = max(0, (len(stream) - 1) // block_size)
    if inputs.shape[0] != n_blocks:
        raise AssertionError("block count must be (N - 1) // block_size; never pad a short tail")
    if n_blocks == 0:
        return
    if inputs.tolist()[0] != stream[:block_size]:
        raise AssertionError("the first block must start at the first token of the stream")
    if (
        targets.tolist()[-1]
        != stream[n_blocks * block_size - block_size + 1 : n_blocks * block_size + 1]
    ):
        raise AssertionError("the last block's targets must end at the stream position it covers")
    if not torch.equal(targets[:, :-1], inputs[:, 1:]) and block_size > 1:
        raise AssertionError("targets must equal inputs shifted left by one inside each block")


def test_documents_given_as_a_dict_raise_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks({}, 2, 0),
        "documents must be a list or tuple; a dict of documents is invalid",
    )


def test_float_token_id_raises_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks([[1.5, 2]], 2, 0),
        "a float token id is not a Python int and must raise ValueError",
    )


def test_negative_one_token_id_raises_value_error():
    _expect_value_error(
        lambda: pack_next_token_blocks([[3, -1]], 2, 0),
        "a token id of -1 is negative and must raise ValueError",
    )


def test_largest_int64_token_id_is_accepted_and_kept():
    inputs, targets = _call([[2**63 - 1]], 1, 0)  # stream (2**63 - 1, 0)
    if inputs.tolist() != [[2**63 - 1]] or targets.tolist() != [[0]]:
        raise AssertionError("the largest int64 id is a valid token and must be packed as is")


def test_token_id_above_int64_raises_even_when_no_block_is_formed():
    _expect_value_error(
        lambda: pack_next_token_blocks([[2**63]], 4, 0),  # stream too short for one block
        "a token id above 2**63 - 1 must raise ValueError even when no block is returned",
    )


def test_tuple_documents_are_accepted_like_list_documents():
    """Contract: each document is a list or a tuple of int token ids."""
    # Stream: (1, 2) + EOS 0 + (3,) + EOS 0 = [1, 2, 0, 3, 0], N = 5, B = (5 - 1) // 2 = 2.
    # Inputs are windows of two tokens starting every two tokens; targets shift them by one.
    inputs, targets = _call([(1, 2), (3,)], 2, 0)
    if inputs.tolist() != [[1, 2], [0, 3]]:
        raise AssertionError("a document given as a tuple must contribute its ids, as a list does")
    if targets.tolist() != [[2, 0], [3, 0]]:
        raise AssertionError("a document given as a tuple must give the same targets as a list")
