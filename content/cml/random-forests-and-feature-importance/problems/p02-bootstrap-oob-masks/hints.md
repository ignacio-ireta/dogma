## Hint 1 — concept

A bootstrap sample is drawn with replacement, so a row can appear several times and some rows never appear. The rows that never appear are the out-of-bag rows for that tree. The whole job is counting: how many times each row was drawn, per tree. Pay attention to the generator too: the protocol uses one generator for all trees, consumed in tree order, so the trees must not each reseed it.

## Hint 2 — formula / approach

For each tree, draw n indices from {0, ..., n−1} with replacement, then count how often each index appears. The counts sum to n for every tree. The out-of-bag mask is the set of rows whose count is zero. Validate first: n and n_trees are integers of at least 1 (booleans are not integers here), and seed is a non-negative integer.

## Hint 3 — code shape

Create `rng = np.random.default_rng(seed)` once, before the loop. Allocate an integer array of shape (n_trees, n). In a loop over trees, draw the indices with `rng.integers(0, n, size=n)` and store their counts in that tree's row, for example with `np.bincount(draws, minlength=n)`. After the loop, the mask is the array of counts compared with zero. Return the pair in the order the contract states.
