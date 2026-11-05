# Round a float to a low-precision format

Implement `round_to_format(x, exponent_bits, mantissa_bits, bias)`. It takes a Python `float` and
returns the value of the given binary format that is nearest to `x`, as a Python `float`. The
format has the same layout and the same rules as in the decode problem: one sign bit, an exponent
field with `exponent_bits` bits and a bias, and a mantissa field with `mantissa_bits` bits.

The rounding rule is **round to nearest, ties to even**. When `x` lies exactly halfway between
two representable values, pick the one whose mantissa field's last bit is 0.

The representable values are:

- normal numbers: `(1 + M / 2**m) * 2**(E - bias)` for `1 <= E <= 2**e - 2`;
- subnormal numbers: `(M / 2**m) * 2**(1 - bias)` for `E = 0`, spaced `2**(1 - bias - m)` apart;
- infinities, which are not values a finite input rounds to unless it overflows.

## Contract

- `x`: `float`. NaN and infinities are allowed.
- `exponent_bits`: `int`, `2 <= exponent_bits <= 8`.
- `mantissa_bits`: `int`, `0 <= mantissa_bits <= 23`.
- `bias`: `int`, `0 <= bias < 2**exponent_bits`.
- Returns: `float`.
  - NaN returns NaN. `+inf` and `-inf` return themselves.
  - A zero or a value that rounds to zero keeps the sign of `x`, so `-0.0` stays `-0.0`.
  - Overflow: if the rounded magnitude would be larger than the largest finite value of the
    format, the result is the infinity with the sign of `x`. This includes values at or above the
    midpoint between the largest finite value and the next power of two.
- Raises `ValueError` when a format parameter is out of range or is a `bool`.
- Must not mutate inputs.

## Examples

```python
>>> round_to_format(1 + 2**-8, 8, 7, 127)      # bf16: a tie between 1 and 1 + 2**-7; 1 is even
1.0
>>> round_to_format(1 + 3 * 2**-8, 8, 7, 127)   # bf16: a tie between 1 + 2**-7 (odd) and 1 + 2**-6 (even)
1.015625
>>> round_to_format(65519.0, 5, 10, 15)         # fp16: below the midpoint 65520, so it rounds down
65504.0
>>> round_to_format(65520.0, 5, 10, 15)         # fp16: exactly the midpoint, ties to the even value 2**16
inf
>>> round_to_format(-(2.0**-26), 5, 10, 15)     # fp16: below half the smallest subnormal, so -0.0
-0.0
>>> round_to_format(3 * 2.0**-25, 5, 10, 15)    # fp16: a tie between 2**-24 (odd) and 2 * 2**-24 (even)
1.1920928955078125e-07
```

## Constraints

- Allowed: the standard library, for example `math.frexp`, `math.ldexp`, `math.copysign`,
  `math.isnan`, `math.isinf`, `math.inf`, `math.nan` and the built-in `round`. Python's `round` on
  a float returns the nearest integer with ties to even, which is useful here.
- Not allowed: `numpy` and `struct`. Their casts round to the format for you, and the tests use
  them as the reference.

## Done looks like

All public tests pass. Then Submit runs the hidden tests. They cover: agreement with IEEE half
precision on seeded random inputs and on exact ties; agreement with bfloat16 rounding on values
that a single-precision number can hold exactly; a brute-force check of small custom formats,
where every representable value is listed and the nearest one is chosen with ties to even;
overflow at and just below the midpoint; underflow to a signed zero and rounding up into the
smallest normal number; NaN, infinities and signed zero; idempotence (rounding a rounded value does
not change it) and the relative error bound; and the `ValueError` rules.
