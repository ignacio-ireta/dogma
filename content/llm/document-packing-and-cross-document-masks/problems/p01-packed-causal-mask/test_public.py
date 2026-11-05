# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
import numpy as np
import pytest
from submission import packed_causal_mask, restart_positions


def test_two_documents_block_the_cross_document_key():
    np.testing.assert_array_equal(
        packed_causal_mask([0, 0, 1]),
        np.array(
            [[True, False, False], [True, True, False], [False, False, True]],
        ),
        err_msg="mask must allow only same-document keys at or before the query",
    )


def test_restart_positions_start_again_at_each_document():
    np.testing.assert_array_equal(
        restart_positions([0, 0, 1, 1, 1]),
        [0, 1, 0, 1, 2],
        err_msg="position ids must restart at 0 at the first token of each document",
    )


def test_empty_row_raises_value_error():
    with pytest.raises(ValueError):
        packed_causal_mask([])
