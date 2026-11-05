## Hint 1 — concept
A word has three fields: a sign bit, an exponent field and a mantissa field. The exponent field alone
decides which case you are in: all zeros (subnormal), all ones (infinity or NaN), or anything else
(normal). Read the fields with shifts and masks before you do any arithmetic.

## Hint 2 — formula / approach
Normal: (1 + M / 2**m) * 2**(E - b). Subnormal: (M / 2**m) * 2**(1 - b). All ones in the exponent:
infinity when M is zero, NaN otherwise. Write each case as an integer significand times a power of two,
so the scaling is exact.

## Hint 3 — code shape
Extract `mantissa` with a mask and `exponent` with a shift and a mask. Check the all-ones exponent first,
then the zero exponent, then the normal case. In each branch compute an integer significand and an
integer power of two, and combine them with `math.ldexp`. Apply the sign last, so a set sign bit on
zero gives -0.0.
