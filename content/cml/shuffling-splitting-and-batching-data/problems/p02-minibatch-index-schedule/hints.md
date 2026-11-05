## Hint 1 — concept

An epoch is one pass in which every row is visited once, in a random order. The order is drawn
from a generator, so the seed fixes the whole schedule. The batch is only a way of cutting that
order into pieces.

## Hint 2 — formula / approach

Create one generator before the epoch loop. Inside the loop, draw one permutation of `n` indices,
then cut it into slices of `batch_size` with `range(0, n, batch_size)`. The last slice can be
shorter; `drop_last` decides whether it is kept.

## Hint 3 — code shape

Validate the three integers first. Then `rng = np.random.default_rng(seed)`, and for each epoch
`perm = rng.permutation(n)` followed by a loop over `range(0, n, batch_size)`. Append each slice to
a list for the epoch, and append that list to the result. Return the result after the epoch loop.
