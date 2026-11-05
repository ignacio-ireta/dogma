"""Public tests: the examples from problem.md and the reserve validation rule."""

from __future__ import annotations

import pytest
from submission import min_gpu_count

PARAMS = 70_000_000_000
CFG = {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128}
GPU = 80 * 2**30
WORKSPACE = 4 * 2**30


def test_thirty_two_sequences_need_four_gpus() -> None:
    got = min_gpu_count(
        PARAMS, CFG, 32, 4096, 2, GPU, workspace_bytes=WORKSPACE, reserve_percent=10
    )
    assert got == 4, (
        "wrong degree: one GPU cannot hold the weights and two cannot hold the cache, so the "
        "answer is 4; check the KV split by KV heads and the usable memory after the reserve"
    )


def test_one_sequence_fits_on_two_gpus() -> None:
    got = min_gpu_count(PARAMS, CFG, 1, 4096, 2, GPU, workspace_bytes=WORKSPACE, reserve_percent=10)
    assert got == 2, (
        "one sequence fits at degree 2: the weights per GPU are 70e9 bytes, and the total is "
        "74,966,055,936 bytes, under the usable memory"
    )


def test_no_allowed_degree_fits_returns_zero() -> None:
    got = min_gpu_count(
        PARAMS, CFG, 512, 32768, 2, GPU, workspace_bytes=WORKSPACE, reserve_percent=10
    )
    assert got == 0, "no degree among 1, 2, 4 and 8 holds this cache: return 0, not a larger degree"


def test_reserve_of_one_hundred_percent_raises_value_error() -> None:
    try:
        min_gpu_count(PARAMS, CFG, 1, 4096, 2, GPU, workspace_bytes=WORKSPACE, reserve_percent=100)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a reserve of 100 percent must raise ValueError, not {type(exc).__name__}")
    pytest.fail(
        "a reserve of 100 percent must raise ValueError: reserve_percent must be at most 99"
    )
