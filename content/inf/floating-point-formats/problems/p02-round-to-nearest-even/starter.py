from __future__ import annotations


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
    raise NotImplementedError
