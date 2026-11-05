## Hint 1 — concept

Under the null hypothesis the labels "a" and "b" carry no information, so every way of splitting the pooled
values into two groups of the original sizes is equally likely. The p-value is the fraction of splits whose
difference is at least as large as the one you observed. Random shuffles estimate that fraction.

## Hint 2 — formula / approach

Compute `observed = |mean(a) - mean(b)|`. For each of `n_permutations` shuffles of the pooled sample, split
at `len(a)`, compute the absolute difference of the two means, and count it when it is at least `observed`
(allowing a tiny relative slack for float ties). Then `p = (count + 1) / (n_permutations + 1)`.

## Hint 3 — code shape

`rng = np.random.default_rng(seed)`, a loop of `rng.permutation(pooled)`, and a `count` that you increment
when the difference passes the threshold. Validate both groups first: 1-D, non-empty, finite, and
`n_permutations >= 1`. Return `(count + 1) / (n_permutations + 1)` as a float.
