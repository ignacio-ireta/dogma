# Hidden tests for decode_float_bits. Oracles are numpy views of the same bit patterns (IEEE half
# and single) and a hand-checked custom format. Failure messages name the kind of mistake and never
# show hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import decode_float_bits


def _decode_or_report(bits: int, e: int, m: int, b: int) -> tuple[float | None, str | None]:
    """Return (value, None), or (None, description) when the call raised."""
    try:
        return decode_float_bits(bits, e, m, b), None
    except ValueError as exc:
        return None, f"raised ValueError on a valid pattern ({exc})"
    except Exception as exc:  # reported after the try block, never raised inside it
        return None, f"raised {type(exc).__name__} on a valid pattern"


def _assert_all_match(values: list[float | None], want: np.ndarray, kind: str) -> None:
    """Compare decoded values with the reference array, bit for bit, including NaN and -0.0."""
    got = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    want = want.astype(np.float64)
    is_nan = np.isnan(want)
    same_nan = np.isnan(got) == is_nan
    same_value = np.where(is_nan, True, got == want)
    same_sign = np.where(is_nan, True, np.signbit(got) == np.signbit(want))
    assert bool(np.all(same_nan & same_value & same_sign)), (
        f"{kind}: decoded value or sign differs from the IEEE reference"
    )


def _no_errors(errors: list[str | None], kind: str) -> None:
    failures = [e for e in errors if e is not None]
    assert not failures, f"{kind}: {failures[0]}"


def test_every_fp16_pattern_matches_ieee_half_precision():
    # Exhaustive: all 65536 half-precision words. Covers normals, subnormals, both zeros,
    # both infinities and every NaN. An error here usually means the subnormal exponent is wrong
    # (it is 1 - bias, with no implicit 1) or the all-ones exponent is not special.
    patterns = np.arange(2**16, dtype=np.uint32)
    decoded = [_decode_or_report(int(p), 5, 10, 15) for p in patterns]
    _no_errors([err for _, err in decoded], "fp16 decode raised")
    want = patterns.astype(np.uint16).view(np.float16)
    _assert_all_match(
        [val for val, _ in decoded],
        want,
        "fp16 mismatch: check the sign of zero, NaN, infinity and subnormals (exponent 1 - bias)",
    )


def test_seeded_random_fp32_patterns_match_ieee_single_precision():
    rng = np.random.default_rng(20240)
    patterns = rng.integers(0, 2**32, size=3000, dtype=np.uint64).astype(np.uint32)
    # Add the edge words explicitly: both zeros, the smallest subnormal, the largest finite
    # value, infinities and a quiet NaN.
    edge = np.array(
        [0x0, 0x80000000, 0x1, 0x7FFFFF, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000],
        dtype=np.uint32,
    )
    patterns = np.concatenate([edge, patterns])
    decoded = [_decode_or_report(int(p), 8, 23, 127) for p in patterns]
    _no_errors([err for _, err in decoded], "fp32 decode raised")
    want = patterns.view(np.float32)
    _assert_all_match(
        [val for val, _ in decoded],
        want,
        "fp32 mismatch: check the implicit leading 1, the exponent bias and the all-ones exponent",
    )


def test_bfloat16_pattern_equals_top_half_of_float32_pattern():
    # bf16 keeps the fp32 sign and exponent and has 7 mantissa bits. Its pattern shifted left by
    # 16 bits is a valid fp32 pattern with the same value.
    patterns = np.arange(2**16, dtype=np.uint32)
    decoded = [_decode_or_report(int(p), 8, 7, 127) for p in patterns]
    _no_errors([err for _, err in decoded], "bfloat16 decode raised")
    want = (patterns << np.uint32(16)).view(np.float32)
    _assert_all_match(
        [val for val, _ in decoded],
        want,
        "bfloat16 mismatch: check the 7-bit mantissa field and the fixed bias of 127",
    )


@settings(max_examples=200, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_property_any_fp32_word_matches_numpy_float32_view(word: int):
    value, err = _decode_or_report(word, 8, 23, 127)
    assert err is None, f"fp32 decode raised: {err}"
    want = np.array([word], dtype=np.uint32).view(np.float32)[0]
    assert value is not None
    kind = "fp32 property mismatch: check the implicit leading 1 and the exponent bias"
    assert (math.isnan(value) and math.isnan(want)) or (
        value == float(want) and math.copysign(1.0, value) == math.copysign(1.0, float(want))
    ), kind


def test_custom_three_bit_format_bias_three_matches_hand_values():
    # Format (e, m, b) = (3, 2, 3): one sign bit, 3 exponent bits, 2 mantissa bits, bias 3.
    # Checked by hand, so it does not depend on any library.
    cases = {
        0b0_001_00: 0.25,  # E = 1: (1 + 0) * 2**(1 - 3)
        0b0_000_01: 0.0625,  # subnormal: (1/4) * 2**(1 - 3)
        0b0_100_01: 2.5,  # E = 4: (1 + 1/4) * 2**(4 - 3)
        0b0_110_11: 14.0,  # largest finite: (1 + 3/4) * 2**(6 - 3)
        0b1_010_10: -0.75,  # sign set: -(1 + 2/4) * 2**(2 - 3)
    }
    for bits, want in cases.items():
        got, err = _decode_or_report(bits, 3, 2, 3)
        assert err is None, f"custom format decode raised: {err}"
        assert got == want, "custom format value wrong: check the bias offset and the implicit 1"


def test_all_ones_exponent_with_nonzero_mantissa_is_nan_not_infinity():
    got, err = _decode_or_report(0b0_111_01, 3, 2, 3)
    assert err is None, f"NaN pattern raised: {err}"
    assert got is not None and math.isnan(got), (
        "NaN pattern decoded as a number: check the mantissa test"
    )


def test_negative_zero_keeps_its_sign_bit():
    zero, err = _decode_or_report(0x0000, 5, 10, 15)
    neg_zero, err2 = _decode_or_report(0x8000, 5, 10, 15)
    assert err is None and err2 is None, "zero pattern raised"
    assert zero == 0.0 and math.copysign(1.0, zero) == 1.0, "positive zero lost its sign"
    assert neg_zero == 0.0 and math.copysign(1.0, neg_zero) == -1.0, (
        "sign bit on zero was dropped: multiply by the sign after the magnitude"
    )


def _expect_value_error(call) -> str | None:
    """Return None when call raises ValueError, otherwise a description of what happened."""
    try:
        call()
    except ValueError:
        return None
    except Exception as exc:  # reported after the try block
        return f"raised {type(exc).__name__} instead of ValueError"
    return "returned a value instead of raising ValueError"


def test_out_of_range_format_parameters_raise_value_error():
    bad_calls = [
        lambda: decode_float_bits(0, 1, 2, 0),  # exponent_bits below 2
        lambda: decode_float_bits(0, 9, 2, 0),  # exponent_bits above 8
        lambda: decode_float_bits(0, 5, 24, 15),  # mantissa_bits above 23
        lambda: decode_float_bits(0, 5, -1, 15),  # mantissa_bits negative
        lambda: decode_float_bits(0, 5, 10, 32),  # bias not below 2**exponent_bits
        lambda: decode_float_bits(0, 5, 10, -1),  # negative bias
    ]
    for call in bad_calls:
        problem = _expect_value_error(call)
        assert problem is None, f"format parameter range not enforced: {problem}"


def test_bool_arguments_and_bits_too_wide_raise_value_error():
    problem = _expect_value_error(lambda: decode_float_bits(True, 5, 10, 15))
    assert problem is None, f"bool argument must be rejected: {problem}"
    problem = _expect_value_error(lambda: decode_float_bits(1 << 16, 5, 10, 15))
    assert problem is None, f"bits wider than the format must be rejected: {problem}"
    problem = _expect_value_error(lambda: decode_float_bits(-1, 5, 10, 15))
    assert problem is None, f"negative bits must be rejected: {problem}"


def test_result_is_a_python_float_not_an_integer():
    value, err = _decode_or_report(0x3C00, 5, 10, 15)
    assert err is None, f"fp16 one raised: {err}"
    assert isinstance(value, float), "result must be a float, not an int or numpy scalar"


def test_zero_mantissa_bits_is_a_valid_format():
    # The contract allows mantissa_bits = 0. With no mantissa field, the normal, subnormal and
    # special exponent rules still apply.
    cases = {
        0b0_001: 0.25,  # E = 1: (1 + 0) * 2**(1 - 3)
        0b0_000: 0.0,  # subnormal with M = 0
        0b0_111: math.inf,  # all-ones exponent with M = 0 is infinity
    }
    for bits, want in cases.items():
        got, err = _decode_or_report(bits, 3, 0, 3)
        assert err is None, f"format with no mantissa bits was rejected: {err}"
        assert got == want, (
            "format with no mantissa bits decoded wrong: check the implicit 1 and the mantissa test"
        )


def test_zero_bias_is_a_valid_format():
    # The contract allows bias = 0, so the exponent field is read without any offset.
    cases = {
        0b0_001_00: 2.0,  # E = 1: (1 + 0) * 2**(1 - 0)
        0b0_000_01: 0.5,  # subnormal: (1/4) * 2**(1 - 0)
    }
    for bits, want in cases.items():
        got, err = _decode_or_report(bits, 3, 2, 0)
        assert err is None, f"zero bias was rejected: {err}"
        assert got == want, "zero bias decoded wrong: check the exponent offset"


def test_two_exponent_bits_is_a_valid_format():
    # The contract allows exponent_bits = 2, the smallest exponent field.
    cases = {
        0b0_01_1: 1.5,  # E = 1, M = 1: (1 + 1/2) * 2**(1 - 1)
        0b0_00_1: 0.5,  # subnormal: (1/2) * 2**(1 - 1)
    }
    for bits, want in cases.items():
        got, err = _decode_or_report(bits, 2, 1, 1)
        assert err is None, f"two exponent bits were rejected: {err}"
        assert got == want, (
            "two-exponent-bit format decoded wrong: check the exponent field and the implicit 1"
        )
