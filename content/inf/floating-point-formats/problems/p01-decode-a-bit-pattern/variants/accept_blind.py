from __future__ import annotations

import math


def _check_int(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def decode_float_bits(bits: int, exponent_bits: int, mantissa_bits: int, bias: int) -> float:
    """Return the real number encoded by ``bits`` in a sign / exponent / mantissa format.

    The word is 1 + exponent_bits + mantissa_bits wide, with the sign in the top bit. Exponent
    field E (exponent_bits wide) and mantissa field M (mantissa_bits wide) decode as:
    normal (0 < E < 2**exponent_bits - 1): (1 + M / 2**mantissa_bits) * 2**(E - bias);
    subnormal (E == 0): (M / 2**mantissa_bits) * 2**(1 - bias);
    E all ones: +-inf if M == 0, otherwise NaN.

    Parameters
    ----------
    bits : int, 0 <= bits < 2**(1 + exponent_bits + mantissa_bits).
    exponent_bits : int, 2 <= exponent_bits <= 8.
    mantissa_bits : int, 0 <= mantissa_bits <= 23.
    bias : int, 0 <= bias < 2**exponent_bits.

    Returns
    -------
    float: the exact value encoded by ``bits``.

    Raises
    ------
    ValueError
        If an argument is out of range, is a bool, or if ``bits`` does not fit the width.
    """
    exponent_bits = _check_int("exponent_bits", exponent_bits, 2, 8)
    mantissa_bits = _check_int("mantissa_bits", mantissa_bits, 0, 23)
    bias = _check_int("bias", bias, 0, (1 << exponent_bits) - 1)
    width = 1 + exponent_bits + mantissa_bits
    bits = _check_int("bits", bits, 0, (1 << width) - 1)

    sign = (bits >> (exponent_bits + mantissa_bits)) & 1
    exp_field = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1)
    man_field = bits & ((1 << mantissa_bits) - 1)
    exp_max = (1 << exponent_bits) - 1

    if exp_field == exp_max:
        magnitude = math.inf if man_field == 0 else math.nan
    elif exp_field == 0:
        # Subnormal: no implicit leading one, exponent is 1 - bias.
        magnitude = math.ldexp(float(man_field), 1 - bias - mantissa_bits)
    else:
        # Normal: (2**m + M) * 2**(E - bias - m) equals (1 + M / 2**m) * 2**(E - bias) exactly.
        magnitude = math.ldexp(
            float((1 << mantissa_bits) + man_field), exp_field - bias - mantissa_bits
        )

    return math.copysign(magnitude, -1.0 if sign else 1.0)
