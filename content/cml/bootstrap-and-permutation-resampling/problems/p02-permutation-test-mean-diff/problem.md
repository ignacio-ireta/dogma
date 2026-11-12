# Permutation test for a difference in means

Implement `permutation_test_mean_diff(a, b, *, n_permutations=5000, seed)`. It returns the two-sided
permutation p-value for the null hypothesis that `a` and `b` come from the same distribution. Under that
null the group labels are exchangeable, so the pooled values can be relabelled at random. The test
statistic is the difference in means.

## Contract

- `a`, `b`: 1-D sequences of finite real numbers (lists or NumPy arrays), each non-empty. Integers are
  allowed.
- `n_permutations`: integer `≥ 1`. Default `5000`.
- `seed`: required integer. For a fixed seed the result is deterministic.
- Algorithm:
  1. `d_obs = |mean(a) − mean(b)|`.
  2. Pool the two samples. Draw `n_permutations` random relabellings (shuffles of the pooled sample).
     In each shuffle the first `len(a)` values form a new group A* and the rest form B*. Set
     `d* = |mean(A*) − mean(B*)|`.
  3. A shuffle counts as *at least as extreme* when `d* ≥ d_obs · (1 − 1e-9)`. The relative tolerance
     absorbs floating-point rounding on exact ties, because two relabellings can give the same
     difference mathematically while their float results differ in the last bit.
  4. `p = (count + 1) / (n_permutations + 1)`, where `count` is the number of shuffles that counted.
- Returns: a float in `(0, 1]`.
- Raises `ValueError` when `a` or `b` is not 1-D, when either is empty, when a value is NaN or infinite,
  or when `n_permutations < 1`.
- Must not mutate its inputs.

## Examples

```python
>>> permutation_test_mean_diff([5, 5, 5], [5, 5, 5], n_permutations=99, seed=0)
1.0                       # every relabelling ties the observed difference of zero
>>> permutation_test_mean_diff([1, 2, 3], [7, 8, 9], n_permutations=9999, seed=0)
0.1...                    # exact two-sided p is 2/20 = 0.1, so the estimate is close to 0.10
>>> permutation_test_mean_diff([], [1.0, 2.0], seed=0)
ValueError: a group is empty
```

## Constraints

- Allowed: NumPy (`np.random.default_rng`, `rng.permutation`, `np.concatenate`, `np.mean`, …).
- Not allowed: `scipy.stats.permutation_test`, which runs the whole test for you.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the p-value with the exact
p-value from enumerating every relabelling of tiny samples, within Monte Carlo error. They cover exact
floating-point ties, near-ties that sit just below the observed difference and are not ties (a loose tie
tolerance counts them), identical groups (p must be exactly 1), the `+1` correction (the p-value has the form
`(count + 1) / (n_permutations + 1)` and is never zero), two-sidedness (the same answer for either group
order), determinism for a fixed seed, invalid arguments, and inputs that are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `iris` (see `docs/DATASETS.md`).
