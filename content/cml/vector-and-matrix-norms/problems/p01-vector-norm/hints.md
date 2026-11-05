## Hint 1 — concept

The p-norm adds up the p-th powers of the magnitudes and takes the 1/p-th root. Two problems can
break that for real inputs. The exponent must be a valid one (p at least 1, or infinity). And the
powers themselves can overflow or underflow when the entries are extreme. Fix the second problem
with a scale factor, which a norm allows you to pull out exactly.

## Hint 2 — formula / approach

For a finite p, `||x||_p = s * (sum (|x_i| / s)**p) ** (1/p)`, where `s = max |x_i|`. Each ratio is
in `[0, 1]`, so no power overflows. Handle these cases first: a zero scale returns 0, and an
infinite p returns the scale directly. Validate `p` with an explicit NaN check, because NaN makes
every comparison false.

## Hint 3 — code shape

Validate `p` and the shape of `x` (one-dimensional and non-empty), then compute the magnitudes with
`np.abs`. Take the maximum as the scale, and branch on the zero scale and on infinite `p`. The
finishing line divides the magnitudes by the scale, raises them to `p`, sums, takes the `1/p` root,
and multiplies the scale back in.
