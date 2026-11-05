# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

from submission import decode_float_bits


def test_fp16_pattern_for_one_decodes_to_one():
    assert decode_float_bits(0x3C00, 5, 10, 15) == 1.0, (
        "normal value wrong: check the implicit leading 1 and the exponent bias"
    )


def test_fp16_smallest_subnormal_is_two_to_the_minus_24():
    # Subnormal: E = 0, so there is no implicit leading 1 and the exponent is 1 - bias.
    assert decode_float_bits(0x0001, 5, 10, 15) == 2.0**-24, (
        "subnormal value wrong: drop the implicit leading 1 when the exponent field is 0"
    )


def test_e5m2_largest_finite_value_is_57344():
    assert decode_float_bits(0b0_11110_11, 5, 2, 15) == 57344.0, (
        "largest finite value wrong: check the exponent bias and the mantissa scale"
    )


def test_all_ones_exponent_with_zero_mantissa_is_infinity_with_sign():
    assert decode_float_bits(0x7C00, 5, 10, 15) == math.inf
    assert decode_float_bits(0xFC00, 5, 10, 15) == -math.inf
