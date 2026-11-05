## Hint 1 — concept
Rounding to a format means choosing the nearest point on a grid. The grid is evenly spaced inside
each binade (the range between two powers of two), and the spacing doubles from one binade to the
next. Below the smallest normal number the spacing stops changing. Ties go to the even mantissa.

## Hint 2 — formula / approach
Find the binade of |x| from its exponent. The spacing in that binade is 2**(max(exp, e_min) - m), where
e_min = 1 - bias. Divide |x| by the spacing, round to the nearest integer with ties to even, and multiply
back. A result above the largest finite value is an overflow to infinity. The largest finite value is
(2 - 2**-m) * 2**(e_max) with e_max = 2**e - 2 - bias.

## Hint 3 — code shape
Handle NaN, the infinities and zero first. For a normal value, get the binade exponent with
`math.frexp`. Take the spacing with `math.ldexp`. Use Python's `round` on the quotient, which breaks
ties to even, then multiply back. Finish with `math.copysign` so the sign survives, and return an infinity
when the rounded value passes the largest finite value.
