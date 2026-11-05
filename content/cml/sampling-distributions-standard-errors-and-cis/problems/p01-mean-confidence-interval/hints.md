## Hint 1 — concept

The interval is the sample mean plus or minus a margin. The margin has two parts: a standard error
that shrinks like 1 / √n, and a multiplier that depends on the confidence level and on how many
observations you have, because σ is estimated from the same sample.

## Hint 2 — formula / approach

`mean ± t · s / √n`. Use `s = np.std(x, ddof=1)`. For `t`, take the `(1 + confidence) / 2` quantile of
Student's t with `n - 1` degrees of freedom. `scipy.stats.t.ppf` gives that quantile. Validate the inputs
before any arithmetic: 1-D, `n ≥ 2`, finite values, and `0 < confidence < 1`.

## Hint 3 — code shape

Convert the input with `np.asarray(x, dtype=np.float64)` so you never write into the caller's array.
Compute `sem = np.std(values, ddof=1) / np.sqrt(n)` and `t_crit = stats.t.ppf(0.5 + confidence / 2, df=n - 1)`.
Return `(mean, mean - t_crit * sem, mean + t_crit * sem)` as floats.
