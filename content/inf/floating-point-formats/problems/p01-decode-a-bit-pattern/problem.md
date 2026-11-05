# Decode a floating-point bit pattern

Implement `decode_float_bits(bits, exponent_bits, mantissa_bits, bias)`. It reads one word of a
binary floating-point format and returns the real number that the word encodes, as a Python
`float`. The word has three fields, from the top bit down: one sign bit, an exponent field of
`exponent_bits` bits, and a mantissa field of `mantissa_bits` bits. The exponent field is stored
with an offset (the bias).

Decode by the rules of the format, not by a library call. The rules are:

- If the exponent field `E` is between 1 and `2**exponent_bits - 2`, the value is a normal
  number: `(1 + M / 2**mantissa_bits) * 2**(E - bias)`, where `M` is the mantissa field as an
  integer. The leading 1 is implicit.
- If `E` is 0, the value is subnormal: `(M / 2**mantissa_bits) * 2**(1 - bias)`. There is no
  implicit leading 1.
- If `E` is `2**exponent_bits - 1` and `M` is 0, the value is infinity. If `M` is not 0, the
  value is NaN.
- The sign bit multiplies the result. A set sign bit on a zero gives `-0.0`.

## Contract

- `bits`: `int`, `0 <= bits < 2**(1 + exponent_bits + mantissa_bits)`. The sign bit is the most
  significant bit of that width.
- `exponent_bits`: `int`, `2 <= exponent_bits <= 8`.
- `mantissa_bits`: `int`, `0 <= mantissa_bits <= 23`.
- `bias`: `int`, `0 <= bias < 2**exponent_bits`.
- Returns: `float`. The value is exact, so no rounding happens here.
- Raises `ValueError` when any argument is out of range, when any argument is a `bool`, or when
  `bits` does not fit in the total width.
- Must not mutate inputs (inputs are ints, so this is automatic; do not keep state between calls).

## Examples

```python
>>> decode_float_bits(0x3C00, 5, 10, 15)    # fp16 pattern 0 01111 0000000000
1.0
>>> decode_float_bits(0x0001, 5, 10, 15)    # smallest positive fp16 subnormal, 2**-24
5.960464477539063e-08
>>> decode_float_bits(0x7BFF, 5, 10, 15)    # largest finite fp16, 65504
65504.0
>>> decode_float_bits(0b0_11110_11, 5, 2, 15)   # E5M2 pattern, largest finite value
57344.0
>>> decode_float_bits(0x7C00, 5, 10, 15)    # exponent all ones, mantissa zero
inf
```

## Constraints

- Allowed: the standard library, for example `math.ldexp`, `math.isnan`, `math.copysign`,
  `math.inf` and `math.nan`. Integer shifts and masks are the intended tool.
- Not allowed: `numpy` and `struct`. Both decode floats from bits for you, and the tests use
  them as the reference.

## Done looks like

All public tests pass. Then Submit runs the hidden tests. They cover: agreement with IEEE half
precision on every one of the 65,536 fp16 patterns (normals, subnormals, zeros, infinities and
NaN); agreement with IEEE single precision on seeded random fp32 patterns and on a property
check; agreement of bfloat16 with the top 16 bits of an fp32 pattern; formats with a non-IEEE
bias and a small mantissa, checked by hand; the sign of zero; the two special exponent cases;
and the `ValueError` rules (range of each parameter, a `bool` argument, and `bits` too large).
