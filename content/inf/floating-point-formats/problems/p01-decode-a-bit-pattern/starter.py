from __future__ import annotations


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
    raise NotImplementedError
