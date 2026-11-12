# Two-proportion z-test

Implement `two_proportion_z_test(x_a, n_a, x_b, n_b)`. Arm A is the control and arm B the treatment. The function
tests whether the two conversion rates differ, using the pooled-variance z statistic and a two-sided p-value. It
returns `(z, p)`.

## Contract

- `x_a`, `x_b`: integer numbers of conversions in arms A and B. `n_a`, `n_b`: integer numbers of users in each
  arm. Python ints and NumPy integers are accepted. Floats and booleans are not counts and raise `ValueError`.
- Requirements: `n_a ≥ 1`, `n_b ≥ 1`, `0 ≤ x_a ≤ n_a`, `0 ≤ x_b ≤ n_b`.
- Formula:

```text
p_a    = x_a / n_a
p_b    = x_b / n_b
p_pool = (x_a + x_b) / (n_a + n_b)              pooled rate, the null hypothesis
z      = (p_b − p_a) / sqrt(p_pool · (1 − p_pool) · (1/n_a + 1/n_b))
p      = 2 · P(Z ≥ |z|),   Z standard normal    two-sided
```

  The sign convention is `z = (p_B − p_A) / SE`. A positive `z` means arm B converted more.
- Returns: a tuple `(z, p)` of two floats.
- Raises `ValueError` when a count is not an integer, when `n_a` or `n_b` is less than 1, when a count is outside
  `[0, n]`, or when the pooled rate is 0 or 1 (the standard error is zero and `z` is undefined).
- Must not mutate its inputs (they are scalars, but do not rely on that).

## Examples

```python
>>> two_proportion_z_test(120, 1000, 90, 1000)
(-2.1882658846, 0.0286502404)     # B converted less than A, so z is negative
>>> two_proportion_z_test(50, 100, 50, 100)
(0.0, 1.0)
>>> two_proportion_z_test(0, 10, 0, 10)
ValueError: pooled rate is 0 or 1: the z statistic is undefined
```

## Constraints

- Allowed: the standard library (`math`), NumPy, and SciPy's normal distribution (for example `scipy.stats.norm.sf`).
- Not allowed: `statsmodels`, including `statsmodels.stats.proportion.proportions_ztest`, which runs the test for you.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare `z` and `p` with
`statsmodels.stats.proportion.proportions_ztest` (pooled variance) on seeded arms of unequal sizes. They check
the sign convention and that swapping the arms flips only the sign, that doubling both arms multiplies `z` by the
square root of two, that a very small p-value is computed in the tail and does not round to zero, that equal rates
give `z = 0` and `p = 1`, that integer counts of any integer type are accepted, that a zero pooled rate and invalid
counts raise `ValueError`, and that non-integer counts are rejected.

The hidden tests draw their bulk inputs from the real dataset `adult` (see `docs/DATASETS.md`).
