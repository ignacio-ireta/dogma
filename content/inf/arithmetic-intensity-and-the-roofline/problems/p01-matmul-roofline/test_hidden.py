"""Hidden tests: an exact rational oracle on seeded random products, the tie and near-tie rules, the
time floor, the attainable rate, exact integer counts, the key set, and validation.

Failure messages name the kind of mistake. They never print hidden shapes or expected values.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Any

import pytest
from submission import matmul_roofline

PEAK = 312_000_000_000_000
BW = 2_000_000_000_000
KEYS = {"flops", "bytes", "intensity", "attainable_flops", "bound", "time_s"}


def _exact(  # noqa: PLR0917 (oracle takes the same six arguments as the contract)
    m: int, n: int, k: int, b: int, peak: int, bw: int
) -> dict[str, Any]:
    """Exact rational roofline: every quantity is a Fraction or an integer until the final float."""
    flops = 2 * m * n * k
    traffic = (m * k + k * n + m * n) * b
    compute = flops * bw >= peak * traffic
    time = max(Fraction(flops, peak), Fraction(traffic, bw))
    attainable = Fraction(peak) if compute else Fraction(flops, traffic) * bw
    return {
        "flops": flops,
        "bytes": traffic,
        "intensity": Fraction(flops, traffic),
        "attainable_flops": attainable,
        "bound": "compute" if compute else "memory",
        "time_s": time,
    }


def _same_int(got: Any, expected: int, mistake: str) -> None:
    if type(got) is not int:
        pytest.fail(f"{mistake}; the count must be a Python int, not {type(got).__name__}")
    if got != expected:
        pytest.fail(mistake)


def _same_float(got: Any, expected: Fraction, mistake: str) -> None:
    if type(got) is not float:
        pytest.fail(f"{mistake}; the value must be a float, not {type(got).__name__}")
    if not math.isclose(got, float(expected), rel_tol=1e-9, abs_tol=0.0):
        pytest.fail(mistake)


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_random_shapes_and_devices_match_exact_rational_roofline() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic shapes, not security
    for _ in range(300):
        m, n, k = rng.randint(1, 4096), rng.randint(1, 4096), rng.randint(1, 4096)
        b = rng.choice([1, 2, 4])
        peak = rng.randint(10**9, 10**15)
        bw = rng.randint(10**9, 10**13)
        want = _exact(m, n, k, b, peak, bw)
        got = matmul_roofline(m, n, k, b, peak, bw)
        _same_int(got["flops"], want["flops"], "flops disagree: the count is 2 * m * n * k")
        _same_int(
            got["bytes"],
            want["bytes"],
            "bytes disagree: count A (m*k), B (k*n) and C (m*n), each at element_bytes",
        )
        _same_float(
            got["intensity"],
            want["intensity"],
            "intensity disagrees: it is FLOPs divided by bytes, not the reverse",
        )
        _same_float(
            got["attainable_flops"],
            want["attainable_flops"],
            "attainable rate disagrees: the peak when compute-bound, else intensity * bandwidth",
        )
        if got["bound"] != want["bound"]:
            pytest.fail(
                "bound disagrees with the ridge test: compare intensity with peak over bandwidth"
            )
        _same_float(
            got["time_s"],
            want["time_s"],
            "time floor disagrees: it is the larger of FLOPs over peak and bytes over bandwidth",
        )


def test_flop_count_is_two_mnk_not_mnk() -> None:
    got = matmul_roofline(3, 5, 7, 2, PEAK, BW)
    _same_int(
        got["flops"],
        210,
        "FLOP count is wrong: a multiply and an add are two FLOPs, so 2 * m * n * k",
    )


def test_output_matrix_is_counted_in_the_traffic() -> None:
    got = matmul_roofline(3, 5, 7, 2, PEAK, BW)
    _same_int(
        got["bytes"],
        (3 * 7 + 7 * 5 + 3 * 5) * 2,
        "traffic is wrong: the output C is written, so its m*n elements belong in the byte count",
    )


def test_exact_tie_at_the_ridge_is_reported_as_compute() -> None:
    got = matmul_roofline(150, 150, 150, 1, 1000, 10)
    if got["bound"] != "compute":
        pytest.fail(
            "a tie at the ridge must be reported as compute-bound: the rule is intensity >= ridge"
        )
    _same_float(
        got["attainable_flops"],
        Fraction(1000),
        "at the tie the attainable rate is the peak: both roofs coincide there",
    )


def test_ridge_a_hair_above_the_intensity_stays_memory_bound() -> None:
    # Intensity is exactly 4.0. The ridge is 4 + 1/(3e18), which rounds to 4.0 in float, so a float
    # comparison would call this a tie. The exact integer comparison says memory-bound.
    got = matmul_roofline(6, 6, 6, 1, 12 * 10**18 + 1, 3 * 10**18)
    if got["bound"] != "memory":
        pytest.fail(
            "a ridge just above the intensity is memory-bound: compare exactly with integers, "
            "since a float ridge rounds this case to a tie"
        )


def test_single_row_product_is_memory_bound_on_a_fast_device() -> None:
    got = matmul_roofline(1, 8192, 8192, 2, PEAK, BW)
    if got["bound"] != "memory":
        pytest.fail(
            "a single-row product reuses each weight once, so it is memory-bound on a fast device"
        )
    _same_float(
        got["time_s"],
        Fraction(got["bytes"], BW),
        "time floor for a memory-bound product is bytes over bandwidth, not FLOPs over peak",
    )


def test_one_by_one_by_one_product_has_minimal_counts() -> None:
    got = matmul_roofline(1, 1, 1, 1, 1, 1)
    _same_int(got["flops"], 2, "a 1 x 1 x 1 product costs 2 FLOPs, not 1")
    _same_int(got["bytes"], 3, "a 1 x 1 x 1 product moves 3 elements: A, B and C")
    _same_float(
        got["intensity"],
        Fraction(2, 3),
        "intensity of the 1 x 1 x 1 product is 2/3 FLOP per byte",
    )
    _same_float(
        got["time_s"], Fraction(3), "time floor of the 1 x 1 x 1 product is max(2 / 1, 3 / 1) = 3 s"
    )


def test_large_dimensions_keep_integer_counts_exact_beyond_float_precision() -> None:
    dim = 2**20
    got = matmul_roofline(dim, dim, dim, 2, PEAK, BW)
    _same_int(
        got["flops"], 2 * 2**60, "FLOP count lost precision or is wrong for very large dimensions"
    )
    _same_int(
        got["bytes"],
        3 * 2**40 * 2,
        "byte count lost precision or is wrong for very large dimensions",
    )
    if not got["flops"] > 2**53:
        pytest.fail("test setup: the FLOP count must exceed 2**53 to test exactness")


def test_time_floor_is_the_larger_of_compute_time_and_memory_time() -> None:
    # Memory-bound here: bytes over bandwidth is the larger of the two times.
    got = matmul_roofline(1, 4096, 4096, 2, PEAK, BW)
    _same_float(
        got["time_s"],
        Fraction(33_570_816, BW),
        "time floor must be the larger term: for this memory-bound product, bytes over bandwidth",
    )


def test_attainable_rate_is_bandwidth_times_intensity_when_memory_bound() -> None:
    got = matmul_roofline(1, 4096, 4096, 2, PEAK, BW)
    _same_float(
        got["attainable_flops"],
        Fraction(33_554_432, 33_570_816) * BW,
        "attainable rate when memory-bound is intensity times bandwidth, not the peak",
    )


def test_returns_exact_key_set_and_value_types() -> None:
    got = matmul_roofline(4, 4, 4, 2, PEAK, BW)
    if set(got) != KEYS:
        pytest.fail(
            "the result must have exactly the six documented keys: flops, bytes, intensity, "
            "attainable_flops, bound, time_s"
        )
    if got["bound"] not in {"memory", "compute"}:
        pytest.fail('bound must be the string "memory" or "compute"')


@pytest.mark.parametrize(
    ("args", "bad_position"),
    [
        ((True, 4, 4, 2, PEAK, BW), "m"),
        ((4, 4.0, 4, 2, PEAK, BW), "n"),
        ((4, 4, "4", 2, PEAK, BW), "k"),
        ((4, 4, 4, None, PEAK, BW), "element_bytes"),
        ((4, 4, 4, 2, PEAK + 0.5, BW), "peak_flops"),
        ((4, 4, 4, 2, PEAK, 2.0e12), "bandwidth"),
    ],
    ids=["bool-m", "float-n", "string-k", "none-element-bytes", "float-peak", "float-bandwidth"],
)
def test_non_int_argument_raises_type_error(args: tuple[Any, ...], bad_position: str) -> None:
    _expect_error(
        TypeError,
        lambda: matmul_roofline(*args),
        f"{bad_position} must be a Python int: bool, float, str and None are rejected",
    )


@pytest.mark.parametrize(
    "args",
    [
        (0, 4, 4, 2, PEAK, BW),
        (4, 0, 4, 2, PEAK, BW),
        (4, 4, -1, 2, PEAK, BW),
        (4, 4, 4, 0, PEAK, BW),
        (4, 4, 4, 2, 0, BW),
        (4, 4, 4, 2, PEAK, 0),
    ],
    ids=["zero-m", "zero-n", "negative-k", "zero-element-bytes", "zero-peak", "zero-bandwidth"],
)
def test_dimension_or_device_below_one_raises_value_error(args: tuple[Any, ...]) -> None:
    _expect_error(
        ValueError,
        lambda: matmul_roofline(*args),
        "every dimension, element width and device rate must be at least 1",
    )
