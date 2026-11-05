# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
from __future__ import annotations

import pytest
import torch
from submission import build_sft_batch


def _rows(tensor: torch.Tensor) -> list[list[int]]:
    return tensor.tolist()


def test_prompt_and_padding_are_masked_and_response_keeps_its_positions():
    input_ids, labels, mask = build_sft_batch([([5, 6], [7, 8, 9]), ([4], [3])], pad_id=0)
    assert _rows(input_ids) == [[5, 6, 7, 8, 9], [4, 3, 0, 0, 0]], (
        "inputs must be prompt + response, right padded"
    )
    assert _rows(labels) == [
        [-100, -100, 7, 8, 9],
        [-100, 3, -100, -100, -100],
    ], "labels must be -100 on prompt and padding, response ids elsewhere; check the prompt offset"
    assert _rows(mask) == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], (
        "attention mask must be 1 on real tokens only"
    )


def test_response_token_equal_to_pad_id_is_still_a_target():
    _, labels, mask = build_sft_batch([([1], [0, 2])], pad_id=0)
    assert _rows(labels) == [[-100, 0, 2]], (
        "a response token equal to pad_id was masked: masks come from lengths, not values"
    )
    assert _rows(mask) == [[1, 1, 1]], "a real token equal to pad_id must keep attention mask 1"


def test_empty_prompt_is_rejected_with_value_error():
    with pytest.raises(ValueError):
        build_sft_batch([([], [3])], pad_id=0)
