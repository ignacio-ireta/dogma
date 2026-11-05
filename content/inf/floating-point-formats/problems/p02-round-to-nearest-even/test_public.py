# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

from submission import round_to_format


def test_bf16_tie_between_one_and_next_value_rounds_to_even_one():
    got = round_to_format(1 + 2**-8, 8, 7, 127)
    assert got == 1.0, "tie rounded to the odd neighbour: ties must go to the even mantissa"


def test_bf16_tie_between_odd_and_even_neighbours_rounds_to_even():
    got = round_to_format(1 + 3 * 2**-8, 8, 7, 127)
    assert got == 1.015625, "tie rounded to the odd neighbour: ties must go to the even mantissa"


def test_fp16_value_at_overflow_midpoint_becomes_infinity():
    assert round_to_format(65520.0, 5, 10, 15) == math.inf, (
        "overflow missed: a value at the midpoint above the largest finite value must become inf"
    )


def test_fp16_just_below_overflow_midpoint_rounds_down():
    assert round_to_format(65519.0, 5, 10, 15) == 65504.0


def test_tiny_negative_value_underflows_to_negative_zero():
    got = round_to_format(-(2.0**-26), 5, 10, 15)
    assert got == 0.0 and math.copysign(1.0, got) == -1.0, (
        "underflow must return a zero with the sign of the input"
    )
