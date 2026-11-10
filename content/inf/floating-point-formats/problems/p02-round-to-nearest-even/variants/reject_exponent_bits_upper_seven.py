from __future__ import annotations

import math


def round_to_format(x: float, exponent_bits: int, mantissa_bits: int, bias: int) -> float:
    """Wrong on purpose: round x to the format, one rule broken."""
    # Validate the format. The ranges keep the grid inside float64.
    for name, value in (
        ("exponent_bits", exponent_bits),
        ("mantissa_bits", mantissa_bits),
        ("bias", bias),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, not a bool or other type")
    if not 2 <= exponent_bits <= 7:
        raise ValueError("exponent_bits must be between 2 and 8")
    if not 0 <= mantissa_bits <= 23:
        raise ValueError("mantissa_bits must be between 0 and 23")
    if not 0 <= bias < 2**exponent_bits:
        raise ValueError("bias must satisfy 0 <= bias < 2**exponent_bits")

    # Special inputs pass through. Zero keeps its sign, so copysign is used at the end.
    if math.isnan(x) or math.isinf(x):
        return x
    if x == 0.0:
        return x

    m = mantissa_bits
    e_min = 1 - bias  # exponent of the smallest normal number
    e_max = (2**exponent_bits - 2) - bias  # exponent of the largest normal number
    # Largest finite value: (2 - 2**-m) * 2**e_max. Written with ldexp so it is exact.
    largest = math.ldexp(2.0 - 2.0**-m, e_max)

    magnitude = abs(x)
    # floor(log2(magnitude)) from frexp: magnitude = f * 2**k with 0.5 <= f < 1.
    exp_x = math.frexp(magnitude)[1] - 1
    # The spacing of representable values is 2**(exp - m) inside each binade. Below the smallest
    # normal, the spacing stays at 2**(e_min - m), which is what makes subnormals evenly spaced.
    spacing = math.ldexp(1.0, max(exp_x, e_min) - m)
    # Dividing by a power of two is exact. Python's round() on a float breaks ties to the even
    # integer, which is exactly the ties-to-even rule.
    steps = round(magnitude / spacing)
    rounded = steps * spacing  # exact: steps is a small integer

    if rounded > largest:
        # Rounding moved past the largest finite value, so the result overflows to infinity.
        return math.copysign(math.inf, x)
    # copysign keeps the sign on results that rounded to zero (signed zero).
    return math.copysign(rounded, x)
