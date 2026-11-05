## Hint 1: concept

Each token may look back only, and only inside its own document. Those are two separate
conditions, and both must hold. Build one boolean matrix for each condition and combine them.

## Hint 2: formula / approach

For the document condition, compare every pair of ids. For the causal condition, compare every
pair of positions with `<=`. Validate the input first, and treat any id that reappears after
another as an error. For restart positions, find where each run begins, carry that start index
forward across the run, and subtract it from the position of each token.

## Hint 3: code shape

Validate, then build two `(T, T)` boolean arrays from broadcast comparisons and return their
`&`. For restart positions, mark the run-start indices and zero elsewhere, take a running maximum
with `np.maximum.accumulate`, and subtract that from `np.arange(T)`. Test a one-document row
against a lower-triangular matrix before you trust the multi-document case.
