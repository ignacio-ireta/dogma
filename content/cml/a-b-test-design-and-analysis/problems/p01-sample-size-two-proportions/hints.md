## Hint 1 — concept

Planning a test is a trade-off between four numbers: the baseline, the effect you care about, the false
positive rate `alpha` and the power `1 − beta`. The sample size needed is the one that makes the difference
between the two rates, divided by its standard error, large enough. The standard error has two versions here:
one under "no difference" and one under "the effect is real".

## Hint 2 — formula / approach

Use `p1 = p_baseline`, `p2 = p1 + mde` and `pbar = (p1 + p2) / 2`. The critical value is
`z_a = Φ⁻¹(1 − alpha / 2)` and the power term is `z_b = Φ⁻¹(power)`. Then
`n = (z_a · sqrt(2·pbar·(1 − pbar)) + z_b · sqrt(p1(1 − p1) + p2(1 − p2)))² / mde²`. Return `ceil(n)`.

## Hint 3 — code shape

Validate first: `0 < p_baseline < 1`, `mde != 0`, `0 < p1 + mde < 1`, `0 < alpha < 1`, `0 < power < 1`, each as a
`not (0 < x < 1)` check so NaN is rejected too. Get the two quantiles from `scipy.stats.norm.ppf`, compute the
numerator, divide by `mde ** 2`, and return `math.ceil(...)`, which is an `int`.
