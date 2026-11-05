"""Public tests: the examples from problem.md and the negative-context edge case."""

from __future__ import annotations

import pytest
from submission import compute_bound_decode_batch

PEAK = 312_000_000_000_000  # 3.12e14 FLOP/s
BW = 2_000_000_000_000  # 2e12 B/s, so the ridge is 156 FLOP/byte
GQA = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
MHA = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
TINY = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}


def test_grouped_query_model_reaches_the_ridge_at_623_sequences() -> None:
    got = compute_bound_decode_batch(7_000_000_000, GQA, 512, 2, peak_flops=PEAK, bandwidth=BW)
    assert got == 623, (
        "the batch threshold is wrong: count the KV bytes with 8 KV heads and 513 tokens per "
        "sequence, and solve the linear inequality for the smallest whole batch"
    )


def test_multi_head_model_at_long_context_never_becomes_compute_bound() -> None:
    got = compute_bound_decode_batch(7_000_000_000, MHA, 512, 2, peak_flops=PEAK, bandwidth=BW)
    assert got == 0, (
        "with 32 KV heads the cache traffic per sequence outgrows the FLOPs, so no batch "
        "reaches the ridge: return 0"
    )


def test_model_already_compute_bound_at_batch_one_returns_one() -> None:
    assert compute_bound_decode_batch(1_000_000, TINY, 0, 1, peak_flops=1, bandwidth=1) == 1, (
        "a batch of one that already reaches the ridge is the answer: return 1, not 0"
    )


def test_negative_context_raises_value_error() -> None:
    try:
        compute_bound_decode_batch(7_000_000_000, GQA, -1, 2, peak_flops=PEAK, bandwidth=BW)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative context must raise ValueError, not {type(exc).__name__}")
    pytest.fail("a negative context must raise ValueError: context_len must be at least 0")
