from __future__ import annotations

import math
from fractions import Fraction


def _check_int(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def round_to_format(x: float, exponent_bits: int, mantissa_bits: int, bias: int) -> float:
    """Round ``x`` to the nearest value of a sign / exponent / mantissa format.

    Rounding is to nearest, ties to the even mantissa. Normal values are
    (1 + M / 2**m) * 2**(E - bias) for 1 <= E <= 2**e - 2. Subnormal values are
    (M / 2**m) * 2**(1 - bias), with spacing 2**(1 - bias - m). Overflow past the largest finite
    value returns an infinity with the sign of ``x``. Underflow returns a zero with the sign of
    ``x``.

    Parameters
    ----------
    x : float, any value including NaN and infinities.
    exponent_bits : int, 2 <= exponent_bits <= 8.
    mantissa_bits : int, 0 <= mantissa_bits <= 23.
    bias : int, 0 <= bias < 2**exponent_bits.

    Returns
    -------
    float: the rounded value in the format.

    Raises
    ------
    ValueError
        If a format parameter is out of range or is a bool.
    """
    exponent_bits = _check_int("exponent_bits", exponent_bits, 2, 8)
    mantissa_bits = _check_int("mantissa_bits", mantissa_bits, 0, 23)
    bias = _check_int("bias", bias, 0, (1 << exponent_bits) - 1)

    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    if x == 0.0:
        return x

    sign = -1.0 if x < 0 else 1.0
    a = abs(Fraction(x))

    # Unbiased exponent e with 2**e <= a < 2**(e+1), computed exactly.
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** e:
        e -= 1
    elif a >= Fraction(2) ** (e + 1):
        e += 1

    # Spacing of representable values near a: normals use the exponent of a, subnormals use
    # the minimum normal exponent 1 - bias.
    quantum = Fraction(2) ** (max(e, 1 - bias) - mantissa_bits)
    rounded = round(a / quantum) * quantum  # Fraction.__round__ ties to even

    max_exp_field = (1 << exponent_bits) - 2
    max_finite = (2 - Fraction(1, 1 << mantissa_bits)) * Fraction(2) ** (max_exp_field - bias)
    if rounded > max_finite:
        return sign * math.inf

    if rounded == 0:
        return math.copysign(0.0, sign)
    return sign * float(rounded)
