from __future__ import annotations

import math


def decode_float_bits(bits: int, exponent_bits: int, mantissa_bits: int, bias: int) -> float:
    """Decode a sign / exponent / mantissa word by the IEEE-style rules (see problem.md)."""
    # Validate the format first. The ranges keep every finite value inside float64, so ldexp
    # below is always exact.
    for name, value in (
        ("bits", bits),
        ("exponent_bits", exponent_bits),
        ("mantissa_bits", mantissa_bits),
        ("bias", bias),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, not a bool or other type")
    if not 2 <= exponent_bits <= 8:
        raise ValueError("exponent_bits must be between 2 and 8")
    if not 0 <= mantissa_bits <= 23:
        raise ValueError("mantissa_bits must be between 0 and 23")
    if not 0 <= bias < 2**exponent_bits:
        raise ValueError("bias must satisfy 0 <= bias < 2**exponent_bits")
    width = 1 + exponent_bits + mantissa_bits
    if not 0 <= bits < 2**width:
        raise ValueError(f"bits must fit in the {width}-bit format")

    # Split the word. The layout from the top bit down is sign | E (exponent) | M (mantissa).
    # Shifts and masks read each field without any floating-point arithmetic.
    mantissa = bits & (2**mantissa_bits - 1)
    exponent = (bits >> mantissa_bits) & (2**exponent_bits - 1)
    sign = -1.0 if (bits >> (width - 1)) else 1.0

    # An all-ones exponent field is reserved: zero mantissa is infinity, anything else is NaN.
    if exponent == 2**exponent_bits - 1:
        return sign * math.inf if mantissa == 0 else math.nan

    if exponent == 0:
        # Subnormal: no implicit leading 1, and the exponent is fixed at 1 - bias.
        # value = M / 2**m * 2**(1 - bias) = M * 2**(1 - bias - m)
        magnitude = math.ldexp(float(mantissa), 1 - bias - mantissa_bits)
    else:
        # Normal: the significand is (2**m + M) / 2**m, so value = (2**m + M) * 2**(E - bias - m).
        # Multiplying by a power of two is exact, so ldexp adds no rounding error.
        significand = 2**mantissa_bits + mantissa
        magnitude = math.ldexp(float(significand), exponent - bias - mantissa_bits)

    # sign * 0.0 keeps the sign of zero, so a set sign bit gives -0.0.
    return sign * magnitude
