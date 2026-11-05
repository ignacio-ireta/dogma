## Hint 1 — concept

A split is a set of random draws followed by bookkeeping. The protocol fixes both: one generator,
one draw per permutation, and a rounding rule for the counts. Most wrong answers change one of
these three things: the rounding, the number of draws, or the order in which classes are visited.

## Hint 2 — formula / approach

Unstratified: the test count is `ceil(test_size * n)`, and one `rng.permutation(n)` gives the rows.
Stratified: compute `test_size * n_j` for each class, floor it, then give the leftover test rows
to the classes with the largest fractional parts, breaking ties by the earlier class. Then draw one
permutation per class, in ascending class order.

## Hint 3 — code shape

Validate first, then `rng = np.random.default_rng(seed)`. For the stratified case, use
`np.unique(labels, return_inverse=True, return_counts=True)`: its classes come out sorted. Loop over
the classes; for each one, `rows = np.flatnonzero(inverse == j)`, `shuffled = rng.permutation(rows)`,
and slice `shuffled` at the class's test count. Sort both index arrays before returning them.
