"""Public tests: the examples from problem.md and the minimum-dimension edge case."""

from __future__ import annotations

import pytest
from submission import matmul_roofline

PEAK = 312_000_000_000_000  # 3.12e14 FLOP/s
BW = 2_000_000_000_000  # 2e12 B/s, so the ridge is 156 FLOP/byte


def test_batch_one_decode_gemv_is_memory_bound_with_exact_counts() -> None:
    got = matmul_roofline(1, 4096, 4096, 2, PEAK, BW)
    assert got["flops"] == 33_554_432, (
        "FLOP count wrong: a matmul costs 2 * m * n * k, not m * n * k"
    )
    assert got["bytes"] == 33_570_816, (
        "byte count wrong: count A, B and the output C, each element at element_bytes"
    )
    assert got["bound"] == "memory", (
        "a batch-1 product has intensity about 1 FLOP/byte, far below the ridge: it is memory-bound"
    )


def test_square_product_is_compute_bound_and_floored_by_compute_time() -> None:
    got = matmul_roofline(4096, 4096, 4096, 2, PEAK, BW)
    assert got["bound"] == "compute", (
        "a large square product has intensity above the ridge: compute-bound"
    )
    assert got["attainable_flops"] == float(PEAK), (
        "a compute-bound product attains the peak FLOP/s, not intensity times bandwidth"
    )
    assert got["time_s"] == pytest.approx(137_438_953_472 / PEAK, rel=1e-9), (
        "time floor wrong: for a compute-bound product it is FLOPs over peak"
    )


def test_intensity_exactly_at_the_ridge_is_reported_as_compute() -> None:
    got = matmul_roofline(150, 150, 150, 1, 1000, 10)
    assert got["intensity"] == 100.0, (
        "intensity wrong: 2 * 150**3 over three 150 x 150 matrices of 1 byte"
    )
    assert got["bound"] == "compute", (
        "at exactly the ridge the tie rule reports compute-bound: use >= for the compute side"
    )


def test_one_by_one_by_one_product_has_minimal_counts() -> None:
    got = matmul_roofline(1, 1, 1, 1, 1, 1)
    assert (got["flops"], got["bytes"]) == (2, 3), (
        "a 1 x 1 x 1 product costs 2 FLOPs and moves 3 one-byte elements"
    )
    assert type(got["bound"]) is str and got["bound"] in {"memory", "compute"}, (
        'bound must be the string "memory" or "compute"'
    )
