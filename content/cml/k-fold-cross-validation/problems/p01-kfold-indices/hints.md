## Hint 1 — concept

Cross-validation asks every position to be held out exactly once. Pick the held-out block for fold
`j`, and let training be everything else. Nothing about the model appears in this function: it only
decides which rows belong to which fold.

## Hint 2 — formula / approach

Get an ordering of the positions: `0..n-1` unshuffled, or one permutation from
`np.random.default_rng(seed)` when shuffling. Split that ordering into `k` consecutive blocks. The first
`n % k` blocks are one longer than the rest. Block `j` is the test set. Training is the complement.

## Hint 3 — code shape

After the two checks (`k < 2` or `k > n`, and a missing seed under shuffle), build
`bounds = np.concatenate([[0], np.cumsum(sizes)])` from the size list. Loop over `j`, sort
`order[bounds[j]:bounds[j+1]]` to get the test block, and use `np.setdiff1d(np.arange(n), test)` for the
training block. Append `(train, test)` to a list and return it.
