## Hint 1 — concept

The estimator asks about subsets, not draws with replacement. Pick `k` of the `n` samples without
replacement. It fails only if every chosen sample is incorrect, so one minus the chance of drawing
only the `n - c` incorrect samples is the answer. Think about what happens when there are fewer
than `k` incorrect samples.

## Hint 2 — formula / approach

The ratio of binomial coefficients telescopes into a product of `k` factors. Each factor is
`1 - c / (n - i)` for `i` from `0` to `k - 1`. Do not evaluate the coefficients themselves. When
the answer is tiny, `1 - product` cancels, so combine the factors as a sum of logarithms and use
`log1p` for each term and `expm1` at the end.

## Hint 3 — code shape

Validate the three integers first, then handle the two degenerate cases before any arithmetic: no
correct sample returns zero, and too few incorrect samples returns one. Otherwise accumulate
`math.log1p(-c / (n - i))` over `i` in a loop of length `k`, and return the negation of
`math.expm1` applied to that sum.
