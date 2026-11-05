# Discrete moments

Implement `discrete_moments(values, probs)`, which returns the mean, variance and standard deviation
of a finite discrete distribution: the random variable `X` takes `values[i]` with probability
`probs[i]`. The variance is the probability-weighted average of the squared distance from the mean,
$\operatorname{Var}(X) = \sum_i p_i (x_i - \mu)^2$. Compute it that way, not as $\mathbb{E}[X^2] - \mu^2$,
because the second form loses precision when the values sit far from zero.

## Contract

- `values`: 1-D sequence of $n \ge 1$ finite real numbers (list or NumPy array; integers allowed).
- `probs`: 1-D sequence of length $n$. Every entry is finite and non-negative, and the sum is within
  $10^{-9}$ of 1.
- Returns: a tuple `(mean, variance, std)` of three floats, where `mean` $= \sum_i p_i x_i$,
  `variance` $= \sum_i p_i (x_i - \mathrm{mean})^2$ and `std` $= \sqrt{\text{variance}}$.
- Raises `ValueError` when `values` or `probs` is not 1-D, when their lengths differ, when either is
  empty, when a value or probability is NaN or infinite, when a probability is negative, or when the
  probabilities do not sum to 1 within $10^{-9}$.
- Must not mutate its inputs.

## Examples

```python
>>> discrete_moments([1, 2, 3, 4, 5, 6], [1/6] * 6)     # a fair die
(3.5, 2.9167, 1.7078)                                  # variance 35/12, shown to 4 places
>>> discrete_moments([0, 1], [0.7, 0.3])                 # a biased coin
(0.3, 0.21, 0.4583)                                    # variance p(1 - p)
>>> discrete_moments([5], [1])                           # a certain outcome
(5.0, 0.0, 0.0)
>>> discrete_moments([0, 1], [0.6, 0.3])
ValueError: probabilities must sum to 1 within 1e-9
```

## Constraints

- Allowed: NumPy array operations (`np.asarray`, `np.sum`, `np.isfinite`, `np.sqrt`, `@`, ...) and
  the `math` module.
- Not allowed: `scipy`, `numpy.var` and `numpy.std`. Compute the weighted mean and variance yourself,
  so the probabilities are used on purpose.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent scipy distribution on random inputs; a variance that stays accurate when every value is
shifted by about $10^9$; the probability checks, including a sum that is 1 only up to float rounding
(accepted) and sums off by a small or a real amount (rejected); NaN, infinite, negative, mismatched, empty and
2-D input raising `ValueError`; a single certain outcome; the shift and scale rules for $aX+b$;
integer input; and inputs left unchanged.
