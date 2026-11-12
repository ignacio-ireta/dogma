# Sample size for two proportions

Implement `sample_size_two_proportions(p_baseline, mde, *, alpha=0.05, power=0.8)`. It returns the number of
users needed in **each** arm of a two-arm A/B test to detect an absolute change of `mde` in a conversion rate.
The test is two-sided, uses the normal approximation, and has significance level `alpha` and power `power`.

## Contract

- `p_baseline`: the control rate `p1`, a float strictly between 0 and 1.
- `mde`: the minimum detectable effect as an absolute difference in rate. It must be nonzero and may be
  negative. The treatment rate is `p2 = p_baseline + mde`.
- `alpha`: two-sided significance level, strictly between 0 and 1. Default `0.05`.
- `power`: the target probability of detecting the effect, strictly between 0 and 1. Default `0.8`.
- The formula (`Φ⁻¹` is the standard normal quantile):

```text
p1   = p_baseline
p2   = p_baseline + mde
pbar = (p1 + p2) / 2                       pooled rate, used under H0 (no difference)
z_a  = Φ⁻¹(1 − alpha / 2)                   two-sided critical value
z_b  = Φ⁻¹(power)                           power = 1 − beta
n    = ( z_a · sqrt(2 · pbar · (1 − pbar))
       + z_b · sqrt(p1 · (1 − p1) + p2 · (1 − p2)) )²  /  (p2 − p1)²
```

  The first term uses the pooled variance (the null hypothesis), and the second uses the unpooled variances
  of the two rates (the alternative hypothesis).
- Returns: `ceil(n)`, a Python `int`, the users required in each arm.
- Raises `ValueError` when `p_baseline` is not strictly between 0 and 1, when `mde = 0`, when `p2` is not
  strictly between 0 and 1, when `alpha` is not strictly between 0 and 1, or when `power` is not strictly
  between 0 and 1.

## Examples

```python
>>> sample_size_two_proportions(0.10, 0.02)
3841
>>> sample_size_two_proportions(0.30, 0.05)
1377
>>> sample_size_two_proportions(0.10, 0.02, alpha=0.01, power=0.9)
7281
>>> sample_size_two_proportions(0.98, 0.05)
ValueError: p_baseline + mde must stay strictly between 0 and 1
```

## Constraints

- Allowed: the standard library (`math`), NumPy, and SciPy's normal distribution (for example `scipy.stats.norm.ppf`).
- Not allowed: `statsmodels`, including its power and proportion modules. The point of the task is to write the
  formula yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the integer with an independent
implementation of the formula over a grid of baselines, effects (including a negative one), `alpha` and `power`,
chosen so that rounding up and rounding to nearest give different answers. They check that a simulation at the
returned size reaches the target power, that more power, a smaller effect or a smaller `alpha` needs more users,
that the return value is an integer, that a very small effect returns a large integer, and that every invalid
argument raises `ValueError`.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `online_retail` and
`spector` (see `docs/DATASETS.md`).
