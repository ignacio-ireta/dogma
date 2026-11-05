# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import torch
from submission import pack_next_token_blocks


def _as_lists(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[list[list[int]], list[list[int]]]:
    inputs, targets = pair
    return inputs.tolist(), targets.tolist()


def test_two_documents_pack_into_two_overlapping_blocks():
    """Each document ends with EOS; blocks share one token; targets are inputs shifted by one."""
    inputs, targets = _as_lists(pack_next_token_blocks([[1, 2], [3]], block_size=2, eos_id=0))
    assert inputs == [[1, 2], [0, 3]], "inputs must be consecutive windows of the EOS-joined stream"
    assert targets == [[2, 0], [3, 0]], "targets must be the inputs shifted left by one token"


def test_leftover_tokens_are_dropped_not_padded():
    """The final EOS does not fill a full block with T = 3, so it is dropped."""
    inputs, targets = _as_lists(pack_next_token_blocks([[7], [8, 9]], block_size=3, eos_id=0))
    assert inputs == [[7, 0, 8]], "a short tail must be dropped, not turned into a padded block"
    assert targets == [[0, 8, 9]], (
        "the targets of the one full block must come from the same window"
    )


def test_empty_document_list_gives_zero_blocks():
    """Edge case: no documents means no stream and no blocks, with the requested width."""
    inputs, targets = pack_next_token_blocks([], block_size=4, eos_id=0)
    assert tuple(inputs.shape) == (0, 4), "an empty corpus must return shape (0, block_size)"
    assert tuple(targets.shape) == (0, 4), "targets must have the same empty shape as inputs"
