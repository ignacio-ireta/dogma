# Unbiased pass@k from n samples

A code or reasoning model is sampled `n` times for one problem, and `c` of the samples pass the
tests. The naive estimate `1 - (1 - c/n)^k` is biased. The unbiased estimator asks: if I drew `k`
samples **without replacement** from the `n` I generated, what is the probability that at least one
of them is correct? Implement `pass_at_k(n, c, k)` for that quantity.

## Contract

- `n`: `int`, `n >= 1`. The number of generated samples.
- `c`: `int`, `0 <= c <= n`. The number that pass.
- `k`: `int`, `1 <= k <= n`. The budget of samples a user would inspect.
- Returns `float` in `[0, 1]`:
  `pass_at_k = 1 - C(n - c, k) / C(n, k)`, where `C` is the binomial coefficient.
- **Degenerate cases.** `c = 0` returns `0.0`. When `n - c < k`, every `k`-subset contains a correct
  sample, so the result is `1.0` exactly.
- **Accuracy.** The result must have relative error at most `1e-9` wherever the true value is
  positive, even when it is tiny. For example, `pass_at_k(10**12, 1, 1)` is `1e-12` to nine digits.
  Absolute error near `1e-16` is not enough for a value of `1e-12`.
- **Range.** Large `n` must work. Intermediate binomial coefficients can be far beyond the range of a
  float, so do not build them as floats.
- Raises `TypeError` when `n`, `c` or `k` is not an `int` (`bool` is not accepted). Raises `ValueError`
  when `n < 1`, `c` is outside `[0, n]`, or `k` is outside `[1, n]`.
- Must not mutate inputs.

## Examples

```python
pass_at_k(5, 2, 1)  # -> 0.4        # equals c / n when k = 1
pass_at_k(5, 2, 2)  # -> 0.7        # 1 - C(3, 2) / C(5, 2) = 1 - 3/10
```

```python
pass_at_k(4, 3, 2)  # -> 1.0        # n - c = 1 < k = 2: any 2 of 4 samples include a correct one
pass_at_k(5, 0, 3)  # -> 0.0        # no sample passed
```

## Constraints

Allowed: Python standard library, including `math.log1p`, `math.expm1` and `math.log`. Not allowed:
`math.comb` (it is the reference for the hidden tests, and it is the shortcut the contract is
written to avoid), `numpy` and `scipy`. The function must return a plain `float`, not an exact
`Fraction`.

## Done looks like

The public examples pass, plus the degenerate and invalid-argument cases. Hidden tests compare
against the exact binomial ratio on seeded small inputs, check the boundary where `n - c = k`, test
large `n` with an exact known value, and test tiny results at relative accuracy. They also check
that the value never decreases as `k` grows, and that every invalid-argument family is rejected.
