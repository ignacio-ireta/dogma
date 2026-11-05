# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import pad_and_truncate


def _as_lists(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[list[list[int]], list[list[int]]]:
    ids, mask = pair
    return ids.tolist(), mask.tolist()


def test_right_padding_puts_pads_after_the_tokens():
    """Right padding appends pads; the mask is 1 only on the real tokens."""
    ids, mask = _as_lists(pad_and_truncate([[5, 6, 7], [8]], pad_id=0, max_length=4))
    assert ids == [[5, 6, 7, 0], [8, 0, 0, 0]], "right padding must place pads after the tokens"
    assert mask == [[1, 1, 1, 0], [1, 0, 0, 0]], "attention mask must be 1 on real tokens only"


def test_left_padding_puts_pads_before_the_tokens():
    """Left padding prepends pads so every row ends in its last real token."""
    ids, mask = _as_lists(
        pad_and_truncate([[5, 6, 7], [8]], pad_id=0, max_length=4, padding_side="left")
    )
    assert ids == [[0, 5, 6, 7], [0, 0, 0, 8]], "left padding must place pads before the tokens"
    assert mask == [[0, 1, 1, 1], [0, 0, 0, 1]], "left-padded mask must be 1 on the last k columns"


def test_left_truncation_keeps_the_last_tokens():
    """With truncation_side='left' the most recent tokens survive."""
    ids, mask = _as_lists(
        pad_and_truncate([[1, 2, 3, 4, 5]], pad_id=9, max_length=3, truncation_side="left")
    )
    assert ids == [[3, 4, 5]], "left truncation must keep the last max_length tokens"
    assert mask == [[1, 1, 1]], "a full row needs no padding, so the mask is all ones"


def test_real_zero_token_is_marked_real_when_pad_id_is_zero():
    """Edge case: the mask follows length, so a real token equal to pad_id stays unmasked."""
    ids, mask = _as_lists(pad_and_truncate([[0, 4]], pad_id=0, max_length=3))
    assert ids == [[0, 4, 0]], "padding must fill the column after the kept tokens"
    assert mask == [[1, 1, 0]], "mask must be by length: a real 0 token is still marked 1"
