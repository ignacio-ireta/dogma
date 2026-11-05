"""Public tests: the examples from problem.md and the empty-prompt edge case."""

from __future__ import annotations

import pytest
from submission import serving_step_times

PEAK = 312_000_000_000_000  # 3.12e14 FLOP/s
BW = 2_000_000_000_000  # 2e12 B/s
PARAMS = 7_000_000_000
MHA = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}


def test_prefill_of_a_long_prompt_is_compute_bound() -> None:
    got = serving_step_times(PARAMS, MHA, 512, 512, 1, 2, peak_flops=PEAK, bandwidth=BW)
    assert got["prefill_s"] == pytest.approx(2 * PARAMS * 512 / PEAK, rel=1e-9), (
        "a 512-token prefill is compute-bound: its time is 2 * params * tokens over peak FLOP/s"
    )


def test_batch_one_decode_step_reads_the_weights_once() -> None:
    got = serving_step_times(PARAMS, MHA, 512, 512, 1, 2, peak_flops=PEAK, bandwidth=BW)
    assert got["decode_step_s"] == pytest.approx(7.134479872e-3, rel=1e-9), (
        "batch-1 decode step: memory-bound, so the time is the weight bytes plus the cache "
        "of 513 tokens, over bandwidth; check the KV factor of 2 and the element width"
    )
    assert got["decode_tokens_per_s"] == pytest.approx(1 / 7.134479872e-3, rel=1e-9), (
        "decode throughput is batch divided by the step time"
    )


def test_batching_shares_the_weight_read_across_sequences() -> None:
    got = serving_step_times(PARAMS, MHA, 512, 512, 8, 2, peak_flops=PEAK, bandwidth=BW)
    assert got["decode_step_s"] == pytest.approx(8.075838976e-3, rel=1e-9), (
        "batch-8 decode step: read the weights once for the batch, and the KV of each sequence"
    )


def test_empty_prompt_raises_value_error() -> None:
    try:
        serving_step_times(PARAMS, MHA, 0, 512, 1, 2, peak_flops=PEAK, bandwidth=BW)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an empty prompt must raise ValueError, not {type(exc).__name__}")
    pytest.fail("an empty prompt must raise ValueError: prompt_len must be at least 1")
