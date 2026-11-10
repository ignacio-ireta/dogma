# Hidden tests for round_to_format. Oracles: numpy's IEEE half cast, a bit-level bfloat16 rounding
# on single-precision values, and a brute-force search over every value of small custom formats
# using exact Fractions. Failure messages name the kind of mistake and never show hidden inputs.
from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import round_to_format


def _round_or_report(x: float, e: int, m: int, b: int) -> tuple[float | None, str | None]:
    try:
        return round_to_format(x, e, m, b), None
    except ValueError as exc:
        return None, f"raised ValueError on a valid input ({exc})"
    except Exception as exc:  # reported after the try block, never raised inside it
        return None, f"raised {type(exc).__name__} on a valid input"


def _same_float(got: float, want: float) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


def _assert_all_match(got_values: list[float | None], want: np.ndarray, kind: str) -> None:
    mismatches = 0
    for got, expected in zip(got_values, want.tolist(), strict=True):
        if got is None or not _same_float(got, expected):
            mismatches += 1
    assert mismatches == 0, f"{kind}"


def test_fp16_matches_numpy_half_cast_on_random_inputs():
    # Random magnitudes across the normal, subnormal and overflow ranges of fp16, with both signs.
    rng = np.random.default_rng(7)
    exponents = rng.uniform(-30.0, 18.0, size=4000)
    signs = rng.choice([-1.0, 1.0], size=4000)
    xs = signs * 2.0**exponents * rng.uniform(1.0, 2.0, size=4000)
    decoded = [_round_or_report(float(x), 5, 10, 15) for x in xs]
    failures = [err for _, err in decoded if err is not None]
    assert not failures, f"fp16 rounding raised: {failures[0]}"
    with np.errstate(over="ignore"):
        want = xs.astype(np.float16).astype(np.float64)
    _assert_all_match(
        [val for val, _ in decoded],
        want,
        "fp16 rounding mismatch: check ties to even, subnormal spacing, sign of zero and overflow",
    )


def test_fp16_exact_ties_all_round_to_even_neighbours():
    # Every midpoint 1 + (2k + 1) * 2**-11 in [1, 2) is an exact tie between two fp16 values.
    k = np.arange(1024, dtype=np.float64)
    xs = 1.0 + (2 * k + 1) * 2.0**-11
    decoded = [_round_or_report(float(x), 5, 10, 15) for x in xs]
    failures = [err for _, err in decoded if err is not None]
    assert not failures, f"fp16 tie rounding raised: {failures[0]}"
    with np.errstate(over="ignore"):
        want = xs.astype(np.float16).astype(np.float64)
    _assert_all_match(
        [val for val, _ in decoded],
        want,
        "fp16 tie mismatch: a tie must go to the neighbour whose mantissa's last bit is 0",
    )


def test_bfloat16_matches_bit_level_rounding():
    # bfloat16 is the top 16 bits of fp32. For a float32 input, adding 0x7FFF plus the lowest kept
    # bit and then truncating is round-to-nearest-even on those 16 bits.
    rng = np.random.default_rng(11)
    xs32 = rng.standard_normal(3000).astype(np.float32) * np.float32(10.0) ** rng.integers(
        -30, 30, size=3000
    ).astype(np.float32)
    bits = xs32.view(np.uint32).astype(np.uint64)
    kept_lsb = (bits >> np.uint64(16)) & np.uint64(1)
    rounded_bits = ((bits + np.uint64(0x7FFF) + kept_lsb) >> np.uint64(16)) << np.uint64(16)
    want = rounded_bits.astype(np.uint32).view(np.float32).astype(np.float64)
    decoded = [_round_or_report(float(x), 8, 7, 127) for x in xs32.astype(np.float64)]
    failures = [err for _, err in decoded if err is not None]
    assert not failures, f"bfloat16 rounding raised: {failures[0]}"
    _assert_all_match(
        [val for val, _ in decoded],
        want,
        "bfloat16 mismatch: check the 7-bit mantissa, the tie rule and the fixed bias of 127",
    )


def _all_candidates(e: int, m: int, b: int) -> list[tuple[Fraction, bool]]:
    """Every non-negative value of the format with a flag for an even mantissa (test-local oracle).

    The virtual value 2**(e_max + 1) is included with an even mantissa. Rounding to it is overflow.
    """
    values: list[tuple[Fraction, bool]] = []
    for big_e in range(2**e - 1):
        for frac_bits in range(2**m):
            if big_e == 0:
                value = Fraction(frac_bits, 2**m) * Fraction(2) ** (1 - b)
            else:
                value = (1 + Fraction(frac_bits, 2**m)) * Fraction(2) ** (big_e - b)
            values.append((value, frac_bits % 2 == 0))
    e_max = (2**e - 2) - b
    values.append((Fraction(2) ** (e_max + 1), True))
    return values


def _brute_force_round(x: float, e: int, m: int, b: int, table: list) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    a = Fraction(abs(x))
    best: tuple[Fraction, bool] | None = None
    best_distance: Fraction | None = None
    for value, even in table:
        distance = abs(a - value)
        if (
            best is None
            or distance < best_distance
            or (distance == best_distance and even and not best[1])
        ):
            best, best_distance = (value, even), distance
    assert best is not None, "the brute-force search found no candidate value to round to"
    chosen = best[0]
    e_max = (2**e - 2) - b
    if chosen == Fraction(2) ** (e_max + 1):
        return math.copysign(math.inf, x)
    return math.copysign(float(chosen), x)


def test_small_custom_formats_match_brute_force_nearest_search():
    # Two small formats: (3, 2, 3) and (4, 3, 7). Every value is listed, and the nearest one is
    # found with exact Fractions. Inputs cover the grid, the midpoints between neighbours.
    for e, m, b in [(3, 2, 3), (4, 3, 7)]:
        table = _all_candidates(e, m, b)
        finite = sorted({value for value, _ in table})
        largest = float(finite[-2])  # the last entry is the virtual overflow value
        rng = np.random.default_rng(100 + e)
        samples = list(rng.uniform(-2 * largest, 2 * largest, size=300))
        grid = [float(v) for v in finite]
        midpoints = [float((lo + hi) / 2) for lo, hi in pairwise(finite)]
        for x in samples + grid + midpoints + [-v for v in grid]:
            want = _brute_force_round(float(x), e, m, b, table)
            got, err = _round_or_report(float(x), e, m, b)
            assert err is None, f"small-format rounding raised: {err}"
            assert got is not None and _same_float(got, want), (
                "small custom format rounded to the wrong value: check ties to even and the "
                "overflow midpoint"
            )


def test_bfloat16_and_fp16_overflow_boundaries_are_exact():
    overflow_msg = "overflow rule wrong: check the midpoint above the largest finite value"
    assert round_to_format(65519.99, 5, 10, 15) == 65504.0, overflow_msg
    assert round_to_format(65520.0, 5, 10, 15) == math.inf, overflow_msg
    assert round_to_format(-1e30, 5, 10, 15) == -math.inf, (
        "overflow of a negative value must be -inf"
    )
    bf16_max = (2.0 - 2.0**-7) * 2.0**127
    assert round_to_format(bf16_max, 8, 7, 127) == bf16_max, (
        "largest finite bfloat16 value must be kept"
    )
    # float32's largest value lies above the bfloat16 overflow midpoint, so it rounds to infinity.
    assert round_to_format(3.4028234663852886e38, 8, 7, 127) == math.inf, (
        "bfloat16 overflow rule wrong: values past the overflow midpoint must become inf"
    )


def test_just_below_smallest_normal_rounds_up_to_it():
    smallest_normal = 2.0**-14
    assert round_to_format(smallest_normal - 2.0**-25, 5, 10, 15) == smallest_normal, (
        "subnormal boundary wrong: a midpoint just below the smallest normal must round up to it"
    )
    assert round_to_format(smallest_normal - 2.0**-26, 5, 10, 15) == smallest_normal, (
        "a value a little further below the smallest normal must still round up to it"
    )


def test_nan_and_infinities_pass_through_and_zero_keeps_its_sign():
    assert math.isnan(round_to_format(math.nan, 5, 10, 15)), "NaN must pass through as NaN"
    assert round_to_format(math.inf, 5, 10, 15) == math.inf, (
        "positive infinity must pass through unchanged"
    )
    assert round_to_format(-math.inf, 5, 10, 15) == -math.inf, (
        "negative infinity must pass through unchanged"
    )
    pos = round_to_format(0.0, 5, 10, 15)
    neg = round_to_format(-0.0, 5, 10, 15)
    assert pos == 0.0 and math.copysign(1.0, pos) == 1.0, "positive zero changed sign"
    assert neg == 0.0 and math.copysign(1.0, neg) == -1.0, "negative zero lost its sign"


@settings(max_examples=150, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.floats(min_value=-60000.0, max_value=60000.0, allow_nan=False))
def test_property_rounding_is_idempotent_for_fp16(x: float):
    once, err = _round_or_report(x, 5, 10, 15)
    assert err is None, f"rounding raised: {err}"
    twice, err2 = _round_or_report(once, 5, 10, 15) if once is not None else (None, None)
    assert err2 is None, f"second rounding raised: {err2}"
    assert once is not None and twice is not None and _same_float(twice, once), (
        "rounding is not idempotent: a value already in the format must come back unchanged"
    )


@settings(max_examples=150, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.floats(min_value=2.0**-14, max_value=60000.0, allow_nan=False))
def test_property_relative_error_within_half_epsilon_in_normal_range(x: float):
    got, err = _round_or_report(x, 5, 10, 15)
    assert err is None, f"rounding raised: {err}"
    assert got is not None and math.isfinite(got), "normal-range value overflowed or became NaN"
    bound = 2.0**-11 * abs(x)
    assert abs(got - x) <= bound, (
        "relative error above half the machine epsilon: check the spacing of the binade"
    )


def test_invalid_format_parameters_raise_value_error():
    def _problem(call) -> str | None:
        try:
            call()
        except ValueError:
            return None
        except Exception as exc:  # reported after the try block
            return f"raised {type(exc).__name__} instead of ValueError"
        return "returned a value instead of raising ValueError"

    checks = [
        lambda: round_to_format(1.0, 1, 2, 0),
        lambda: round_to_format(1.0, 9, 2, 0),
        lambda: round_to_format(1.0, 5, 24, 15),
        lambda: round_to_format(1.0, 5, -1, 15),
        lambda: round_to_format(1.0, 5, 10, 32),
        lambda: round_to_format(1.0, 5, 10, -1),
        lambda: round_to_format(1.0, True, 10, 15),
    ]
    for call in checks:
        problem = _problem(call)
        assert problem is None, f"format parameter range not enforced: {problem}"


def test_result_is_a_python_float():
    got, err = _round_or_report(1.5, 8, 7, 127)
    assert err is None, f"rounding raised: {err}"
    assert isinstance(got, float), "result must be a float, not an integer or numpy scalar"


def test_zero_mantissa_bits_is_a_valid_format():
    # The contract allows mantissa_bits = 0. With format (3, 0, 3) the values are 0, 0.25, 0.5,
    # 1, 2, 4 and 8, and neither input below is a tie.
    got, err = _round_or_report(2.9, 3, 0, 3)
    assert err is None, f"format with no mantissa bits was rejected: {err}"
    assert got == 2.0, "format with no mantissa bits rounded to the wrong neighbour"
    got, err = _round_or_report(3.5, 3, 0, 3)
    assert err is None, f"format with no mantissa bits was rejected: {err}"
    assert got == 4.0, "format with no mantissa bits rounded to the wrong neighbour"


def test_zero_bias_is_a_valid_format():
    # The contract allows bias = 0. With format (3, 2, 0) the values from 2 to 3.5 are 2, 2.5,
    # 3 and 3.5, and neither input below is a tie.
    got, err = _round_or_report(2.4, 3, 2, 0)
    assert err is None, f"zero bias was rejected: {err}"
    assert got == 2.5, "zero bias rounded to the wrong neighbour: check the exponent offset"
    got, err = _round_or_report(2.1, 3, 2, 0)
    assert err is None, f"zero bias was rejected: {err}"
    assert got == 2.0, "zero bias rounded to the wrong neighbour: check the exponent offset"


def test_two_exponent_bits_is_a_valid_format():
    # The contract allows exponent_bits = 2. With format (2, 1, 1) the top values are 2 and 3,
    # and neither input below is a tie.
    got, err = _round_or_report(2.4, 2, 1, 1)
    assert err is None, f"two exponent bits were rejected: {err}"
    assert got == 2.0, "two-exponent-bit format rounded to the wrong neighbour"
    got, err = _round_or_report(2.6, 2, 1, 1)
    assert err is None, f"two exponent bits were rejected: {err}"
    assert got == 3.0, "two-exponent-bit format rounded to the wrong neighbour"


def test_twenty_three_mantissa_bits_is_a_valid_format():
    # The contract allows mantissa_bits = 23, the full single-precision field.
    exact = 1.0 + 2.0**-23
    got, err = _round_or_report(exact, 8, 23, 127)
    assert err is None, f"23 mantissa bits were rejected: {err}"
    assert got == exact, "a value on the 23-bit grid must come back unchanged"
    # 1 + 3 * 2**-24 is a tie between 1 + 2**-23 (odd) and 1 + 2 * 2**-23 (even).
    got, err = _round_or_report(1.0 + 3.0 * 2.0**-24, 8, 23, 127)
    assert err is None, f"23 mantissa bits were rejected: {err}"
    assert got == 1.0 + 2.0**-22, "23-bit tie must round to the even neighbour"


def test_bool_format_parameters_raise_value_error():
    # A bool is never a valid format parameter, even when its value is inside the range.
    for call in (
        lambda: round_to_format(1.0, 5, False, 15),
        lambda: round_to_format(1.0, 5, 10, True),
    ):
        try:
            call()
        except ValueError:
            continue
        except Exception as exc:
            raise AssertionError(f"raised {type(exc).__name__} instead of ValueError") from exc
        raise AssertionError("a bool format parameter was accepted: it must raise ValueError")
