"""Public tests: the examples from problem.md and the empty-budget case."""

from __future__ import annotations

from typing import Any

import pytest
from submission import max_batch_size

MHA_7B = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
GQA_WINDOWED = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "sliding_window": 4096}
WEIGHTS_7B_FP16 = 14_000_000_000


def _batch(config: Any, seq_len: int, memory: int, reserve: int = 0) -> int:
    got = max_batch_size(config, seq_len, "fp16", WEIGHTS_7B_FP16, memory, reserve_bytes=reserve)
    if type(got) is not int:
        pytest.fail("return a Python int number of sequences, not a float")
    return got


def test_thirty_three_sequences_fit_on_an_eighty_gib_device() -> None:
    assert _batch(MHA_7B, 4096, 80 * 2**30) == 33, (
        "wrong batch on 80 GiB: subtract the weights first, then count whole 2 GiB sequences "
        "with a floor, not a ceiling"
    )


def test_one_sequence_fits_in_sixteen_gib_and_two_do_not() -> None:
    assert _batch(MHA_7B, 4096, 16 * 2**30) == 1, (
        "wrong batch on 16 GiB: an exact count of whole sequences in the budget left after weights"
    )


def test_reserve_takes_budget_before_the_count() -> None:
    assert _batch(MHA_7B, 4096, 80 * 2**30, reserve=2 * 2**30) == 32, (
        "the reserve must come out of the budget with the weights: one 2 GiB sequence is lost"
    )


def test_sliding_window_sequences_cost_the_window_size() -> None:
    got = max_batch_size(GQA_WINDOWED, 32768, "fp16", WEIGHTS_7B_FP16, 80 * 2**30)
    assert got == 133, (
        "wrong batch with an 8-KV-head window: a sequence stores the window, not the full context"
    )


def test_weights_larger_than_memory_leave_no_batch() -> None:
    assert _batch(MHA_7B, 4096, 10 * 2**30) == 0, (
        "weights larger than memory must give batch 0, not a negative number"
    )


def test_zero_seq_len_is_rejected_with_value_error() -> None:
    try:
        max_batch_size(MHA_7B, 0, "fp16", WEIGHTS_7B_FP16, 80 * 2**30)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"seq_len 0 must raise ValueError, instead it raised {type(exc).__name__}")
    pytest.fail("seq_len 0 must raise ValueError: an empty sequence has no bound on the batch")
