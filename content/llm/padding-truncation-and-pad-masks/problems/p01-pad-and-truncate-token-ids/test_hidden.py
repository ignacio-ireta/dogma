# Hidden tests. Oracles: torch.nn.utils.rnn.pad_sequence (tests may use torch.nn, the learner may
# not) and a plain-Python reference built from list slicing. Messages name the kind of mistake.
from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import pad_and_truncate
from torch.nn.utils.rnn import pad_sequence  # oracle only

TOKEN = st.integers(min_value=0, max_value=50)


def _call(sequences, pad_id, width, trunc="right", side="right"):
    """Run the learner on a valid call. Failures are AssertionErrors that name the mistake."""
    snapshot = [list(row) for row in sequences]
    try:
        out = pad_and_truncate(sequences, pad_id, width, truncation_side=trunc, padding_side=side)
    except Exception as exc:
        raise AssertionError(
            f"valid input raised {type(exc).__name__}: check the signature and the keyword "
            "names truncation_side and padding_side"
        ) from None
    if [list(row) for row in sequences] != snapshot:
        raise AssertionError("input rows were modified; slice or copy instead of changing them")
    if not isinstance(out, tuple) or len(out) != 2 or not all(torch.is_tensor(t) for t in out):
        raise AssertionError("return a tuple of two torch tensors: input_ids and attention_mask")
    ids, mask = out
    rows = len(sequences)
    if tuple(ids.shape) != (rows, width) or tuple(mask.shape) != (rows, width):
        raise AssertionError("both outputs must have shape (B, max_length)")
    if ids.dtype != torch.int64 or mask.dtype != torch.int64:
        raise AssertionError("outputs must be int64: token ids are not floats and must not be cast")
    return ids, mask


def _reference(sequences, pad_id, width, trunc, side):
    """Plain-Python oracle: slice, pad, then mark kept positions by length."""
    ids_rows, mask_rows = [], []
    for seq in sequences:
        kept = list(seq[:width]) if trunc == "right" else list(seq[max(0, len(seq) - width) :])
        n_pad = width - len(kept)
        if side == "right":
            ids_rows.append(kept + [pad_id] * n_pad)
            mask_rows.append([1] * len(kept) + [0] * n_pad)
        else:
            ids_rows.append([pad_id] * n_pad + kept)
            mask_rows.append([0] * n_pad + [1] * len(kept))
    return ids_rows, mask_rows


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


def _ragged(gen: torch.Generator, rows: int, max_len: int, vocab: int) -> list[list[int]]:
    """Seeded ragged batch: row lengths in [0, max_len], ids in [0, vocab)."""
    out = []
    for _ in range(rows):
        n = _int(gen, 0, max_len)
        out.append(torch.randint(0, vocab, (n,), generator=gen).tolist())
    return out


def test_right_padding_matches_pad_sequence_when_nothing_is_truncated():
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        seqs = _ragged(gen, _int(gen, 1, 5), 6, 100)
        seqs = [s if s else [1] for s in seqs]  # pad_sequence needs non-empty rows for the oracle
        width = max(len(s) for s in seqs)
        ids, mask = _call(seqs, 0, width, "right", "right")
        want_ids = pad_sequence([torch.tensor(s) for s in seqs], batch_first=True, padding_value=0)
        lengths = torch.tensor([len(s) for s in seqs])
        want_mask = (torch.arange(width)[None, :] < lengths[:, None]).to(torch.int64)
        if not torch.equal(ids, want_ids):
            raise AssertionError(
                "right padding differs from the reference: pads belong after the tokens"
            )
        if not torch.equal(mask, want_mask):
            raise AssertionError("right-padded mask differs: it must be 1 on the real tokens only")


def test_left_padding_matches_flipped_pad_sequence():
    gen = torch.Generator().manual_seed(1)
    for _ in range(20):
        seqs = _ragged(gen, _int(gen, 1, 5), 6, 100)
        seqs = [s if s else [1] for s in seqs]
        width = max(len(s) for s in seqs)
        ids, mask = _call(seqs, 0, width, "right", "left")
        want_ids = pad_sequence(
            [torch.tensor(s).flip(0) for s in seqs], batch_first=True, padding_value=0
        ).flip(1)
        lengths = torch.tensor([len(s) for s in seqs])
        want_mask = (torch.arange(width)[None, :] >= (width - lengths)[:, None]).to(torch.int64)
        if not torch.equal(ids, want_ids):
            raise AssertionError(
                "left padding differs from the reference: pads must come before the tokens"
            )
        if not torch.equal(mask, want_mask):
            raise AssertionError("left-padded mask must be 1 on the last real-token columns")


def test_random_batches_match_plain_python_reference_for_every_side_choice():
    gen = torch.Generator().manual_seed(2)
    for _ in range(40):
        seqs = _ragged(gen, _int(gen, 0, 5), 9, 30)
        width = _int(gen, 1, 7)  # sometimes shorter than a row, sometimes longer
        pad_id = _int(gen, 0, 29)
        for trunc in ("right", "left"):
            for side in ("right", "left"):
                ids, mask = _call(seqs, pad_id, width, trunc, side)
                want_ids, want_mask = _reference(seqs, pad_id, width, trunc, side)
                if ids.tolist() != want_ids:
                    raise AssertionError(
                        "input_ids differ from the reference: 'right' truncation keeps the first "
                        "max_length tokens, 'left' keeps the last; padding_side places pads"
                    )
                if mask.tolist() != want_mask:
                    raise AssertionError(
                        "attention_mask differs from the reference: mark kept positions by length"
                    )


def test_real_token_equal_to_pad_id_stays_unmasked_in_an_all_zero_row():
    ids, mask = _call([[0, 0, 0]], 0, 4)
    if mask.tolist() != [[1, 1, 1, 0]]:
        raise AssertionError("mask must follow length: a row of real zeros is still real tokens")
    if ids.tolist() != [[0, 0, 0, 0]]:
        raise AssertionError("ids must keep the real zero tokens and pad with pad_id")


def test_empty_row_is_all_padding_with_a_zero_mask():
    ids, mask = _call([[]], 7, 3, "right", "left")
    if ids.tolist() != [[7, 7, 7]]:
        raise AssertionError("an empty row must be filled with pad_id in every column")
    if mask.tolist() != [[0, 0, 0]]:
        raise AssertionError("an empty row has no real tokens, so its mask must be all zeros")


def test_empty_batch_returns_zero_rows_with_the_requested_width():
    ids, mask = _call([], 0, 5)
    if tuple(ids.shape) != (0, 5) or tuple(mask.shape) != (0, 5):
        raise AssertionError("an empty batch must return shape (0, max_length), not an error")


def test_token_ids_above_float32_precision_survive_exactly():
    big = 2**40 + 3  # above 2**24, so a float32 cast would change it
    ids, _ = _call([[big, 7]], 0, 3)
    if ids.tolist() != [[big, 7, 0]]:
        raise AssertionError(
            "token ids changed value: keep them in an int64 tensor, not a float one"
        )


def test_maximum_int64_token_id_is_accepted_and_kept():
    top = 2**63 - 1
    ids, mask = _call([[top]], 0, 2)
    if ids.tolist() != [[top, 0]] or mask.tolist() != [[1, 0]]:
        raise AssertionError("the largest int64 id is a valid token and must be kept as is")


def test_invalid_side_name_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[1]], 0, 2, truncation_side="middle"),
        "a truncation side other than 'left' or 'right' must raise ValueError",
    )
    _expect_value_error(
        lambda: pad_and_truncate([[1]], 0, 2, padding_side="center"),
        "a padding side other than 'left' or 'right' must raise ValueError",
    )


def test_zero_max_length_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[1]], 0, 0),
        "max_length must be at least 1; zero width is invalid",
    )


def test_bool_max_length_is_rejected_even_though_bool_is_an_int():
    _expect_value_error(
        lambda: pad_and_truncate([[1]], 0, True), "a bool max_length must raise ValueError"
    )


def test_negative_token_id_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[1, -2]], 0, 3), "a negative token id must raise ValueError"
    )


def test_token_id_beyond_int64_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[2**63]], 0, 2),
        "a token id above the int64 range must raise ValueError, not overflow",
    )


def test_flat_list_of_ints_instead_of_rows_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([1, 2, 3], 0, 4),  # type: ignore[arg-type]
        "sequences must be a batch of token-id lists; a flat list of ints is invalid",
    )


def test_inputs_are_not_mutated_by_truncation_or_padding():
    rows = [[1, 2, 3, 4, 5], [6]]
    snapshot = [list(r) for r in rows]
    pad_and_truncate(rows, 0, 2, truncation_side="left", padding_side="left")
    if rows != snapshot:
        raise AssertionError("input rows were modified; truncate by slicing a copy")


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    sequences=st.lists(st.lists(TOKEN, max_size=8), max_size=5),
    pad_id=TOKEN,
    width=st.integers(min_value=1, max_value=6),
    trunc=st.sampled_from(["left", "right"]),
    side=st.sampled_from(["left", "right"]),
)
def test_property_mask_sums_to_kept_length_and_pads_hold_pad_id(
    sequences, pad_id, width, trunc, side
):
    ids, mask = _call(sequences, pad_id, width, trunc, side)
    for r, seq in enumerate(sequences):
        kept = min(len(seq), width)
        if int(mask[r].sum()) != kept:
            raise AssertionError("mask row sum must equal min(length, max_length): mask by length")
        pad_cols = mask[r] == 0
        if not bool((ids[r][pad_cols] == pad_id).all()):
            raise AssertionError("every padding column must hold pad_id")
        real = ids[r][mask[r] == 1].tolist()
        expected = list(seq[:width] if trunc == "right" else seq[max(0, len(seq) - width) :])
        if real != expected:
            raise AssertionError(
                "kept tokens must appear in order, from the side named by truncation_side"
            )


def test_sequences_given_as_a_dict_raise_value_error():
    _expect_value_error(
        lambda: pad_and_truncate({}, 0, 2),
        "sequences must be a list or tuple; a dict of rows is invalid",
    )


def test_float_token_id_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[1.5, 2]], 0, 3),
        "a float token id is not a Python int and must raise ValueError",
    )


def test_negative_one_token_id_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[4, -1]], 0, 3),
        "a token id of -1 is negative and must raise ValueError",
    )


def test_negative_pad_id_raises_value_error():
    _expect_value_error(
        lambda: pad_and_truncate([[1]], -1, 2),
        "pad_id must be at least 0; a negative pad_id must raise ValueError",
    )


def test_token_id_above_int64_is_rejected_even_when_truncation_drops_it():
    _expect_value_error(
        lambda: pad_and_truncate([[2**63, 1]], 0, 1, truncation_side="left"),
        "an id above the int64 range must raise ValueError even when truncation drops it",
    )
