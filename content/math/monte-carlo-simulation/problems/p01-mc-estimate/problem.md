# Monte Carlo estimate with a confidence interval

Implement `mc_estimate(f, sample, n, *, seed, confidence=0.95)`. It estimates an expectation
E[f(X)] by simulation. `sample(rng, n)` draws `n` values of X from a NumPy generator `rng`, and `f`
maps the sample to integrand values. The function returns the estimate, its standard error and a
normal-approximation confidence interval.

## Contract

- `f`: a vectorised function. It is applied **once** to the whole sample array `x`, of shape
  `(n,)`, and returns values of shape `(n,)`.
- `sample`: `sample(rng, n)` returns an array of `n` draws of X. It is called **once**, with the
  same `n`, and with `rng = np.random.default_rng(seed)`, created once before the call.
- `n`: an int, `n >= 2`.
- `seed`: an int, keyword-only. It is used only to build `rng`.
- `confidence`: a float strictly between 0 and 1, keyword-only, default `0.95`.
- Returns: a tuple `(estimate, standard_error, (lo, hi))` of Python floats, where
  - `estimate` is the mean of `y = f(x)`;
  - `standard_error` is `s / sqrt(n)`, where `s` is the sample standard deviation of `y` with
    divisor `n - 1`;
  - `(lo, hi)` is `(estimate - z * standard_error, estimate + z * standard_error)`, with
    `z = Phi^-1((1 + confidence) / 2)` the standard normal quantile. For `confidence=0.95`,
    `z` is about 1.959964.
- Raises `ValueError` when `n < 2`, or when `confidence` is not strictly between 0 and 1.
- Must not modify the arrays that `sample` returns.

## Examples

```python
>>> mc_estimate(lambda x: x, lambda rng, n: np.arange(1.0, n + 1), 4, seed=0)
(2.5, 0.6455, (1.2348, 3.7652))   # draws [1, 2, 3, 4]: s = 1.2910 with n - 1, se = s / 2
>>> mc_estimate(lambda x: x**2, lambda rng, n: rng.random(n), 1000, seed=0)
(0.3482, 0.0094, (0.3298, 0.3665))   # the interval covers E[U^2] = 1/3
>>> mc_estimate(lambda x: x, lambda rng, n: rng.random(n), 1, seed=0)
ValueError                        # one draw has no sample standard deviation
```

## Constraints

- Allowed: NumPy, and `statistics.NormalDist` from the standard library, for the normal quantile.
- Not allowed: `scipy`. Compute the mean and the standard error yourself from the sample.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an
independent recomputation on random problems, where the sampler, integrand, sample size, seed and
confidence all vary; an exact hand-computed case with a fixed sample; that `sample` and `f` are each
called once, with the right sizes, from a generator seeded by `seed`; determinism by seed; the
coverage of the interval across many seeds; a closed-form expectation, E[exp(Z)] = e^(1/2); the
standard error falling like 1/sqrt(n); numerical stability at a large offset; a zero-variance
integrand; validation of `n` and `confidence`; the sample array left unchanged in value and order; and the shape of the return value.
