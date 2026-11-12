# Population stability index

Implement `population_stability_index(expected, actual, *, bins=10, eps=1e-6)`. It compares how a feature is
distributed in production (`actual`) with how it was distributed when the model was trained (`expected`), and
returns one number. It is 0 when the two shares agree bin by bin, and it grows as they diverge.

## Contract

- `expected`: 1-D sequence of finite real numbers, non-empty. The reference sample. Integers are allowed.
- `actual`: 1-D sequence of finite real numbers, non-empty. The current sample.
- `bins`: integer >= 2. A bool or a float is rejected.
- `eps`: finite positive float.
- Bin edges come from `expected` only. The interior edges are its quantiles at k / bins for k = 1 .. bins - 1,
  with linear interpolation (NumPy's default method). The outer edges are -inf and +inf, so every value falls in
  some bin. A value goes to the bin whose index equals the number of interior edges strictly below it. A value
  exactly on an edge therefore belongs to the lower bin.
- Shares: p_k = (number of expected values in bin k) / (number of expected values), and q_k likewise for `actual`.
  Every zero share is replaced by `eps`. Nonzero shares are not changed, and nothing is renormalised.
- PSI = sum over k of (q_k - p_k) * ln(q_k / p_k), with the natural logarithm. It is never negative, and it is
  exactly 0 when the shares are identical.
- Repeated interior edges (ties in `expected`) are allowed. They make empty bins, which take `eps` as usual.
- Returns: a Python float.
- Raises `ValueError` when an input is not 1-D, is empty, or has NaN or infinity; when `bins` is not an integer of
  at least 2; or when `eps` is not finite and positive.
- Must not mutate its inputs.

## Examples

```python
>>> population_stability_index([1, 2, 3, 4], [1, 2, 3, 4], bins=2)
0.0
```

```python
>>> population_stability_index([1, 2, 3, 4], [1, 2, 3, 4, 4, 4], bins=2)
0.11552453009332421
```

```python
>>> population_stability_index([1, 2, 3, 4], [1, 1, 1, 1], bins=2)
6.907742156618760
```

In the second example the edge is 2.5. The expected shares are 1/2 and 1/2, and the actual shares are 2/6 and
4/6. The result is (1/6) ln 2. The third example has an empty actual bin, so its share is replaced by eps = 1e-6.

## Constraints

- Allowed: NumPy and the Python standard library.
- Not allowed: `pandas` (`cut` uses equal-width bins by default, and `qcut` rejects repeated edges unless
  told to drop them) and `scipy`. The point of the
  problem is the bin rule and the sum, so both have to be written here.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the result with an explicit,
loop-based implementation of the bin rule and the sum, on seeded samples with several bin counts and unequal sizes.
They check the edge rule: a value exactly on an edge belongs to the lower bin. They check that the bins come from the
expected sample, so a shifted actual sample does not re-bin itself, and that actual values outside the expected range
stay in the outer bins instead of being dropped. They check the eps floor for an empty bin, including a custom eps, a
constant reference, invariance under a common shift and rescaling, and that the result is never negative. The tests
also cover invalid `bins` and `eps`, an empty or non-finite input, a two-dimensional input, integer input, and inputs
that must not be mutated.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).
