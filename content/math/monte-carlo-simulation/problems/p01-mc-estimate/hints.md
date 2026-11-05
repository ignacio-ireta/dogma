## Hint 1 — concept

The estimate is an average over draws. The uncertainty of an average of `n` independent draws is
the sample standard deviation divided by `sqrt(n)`. The interval is the estimate plus or minus a
normal quantile times that error. Check `n` and `confidence` before you draw anything.

## Hint 2 — formula / approach

Draw once: `x = sample(rng, n)` with `rng = np.random.default_rng(seed)`. Apply `f` once to the
whole array. Then `estimate = mean(y)`, `s**2 = sum((y - estimate)**2) / (n - 1)`,
`standard_error = s / sqrt(n)`, and `z = NormalDist().inv_cdf((1 + confidence) / 2)`.

## Hint 3 — code shape

Validate first, with `raise ValueError` for `n < 2` or a `confidence` outside (0, 1). Then build the
generator, call `sample` and `f` once each, and compute the mean first. Sum the squared deviations
from that mean in a second pass. Return a tuple: the estimate, the standard error, and a
two-element tuple `(estimate - z * se, estimate + z * se)`.
